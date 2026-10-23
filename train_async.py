# Modified for COPC.
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    import logging
    from slime.utils.async_checkpoint_policy import resolve_checkpoint_mode

    logger = logging.getLogger(__name__)
    pipeline_checkpoints = resolve_checkpoint_mode(args)
    from slime.utils.async_checkpoint import (
        prepare_resume, restore_pipeline_checkpoint, save_pipeline_checkpoint,
    )
    resume_manifest = prepare_resume(args)
    if resume_manifest is not None and resume_manifest["next_rollout_id"] == args.num_rollout:
        # A completed commit may precede job teardown.
        return
    release_train = args.release_train
    # Keep the actor resident until COPC pre-forward releases it to the critic.
    args._keep_actor_resident = (
        args.train_backend == "megatron"
        and args.advantage_estimator == "copc"
        and args.use_critic
        and args.offload_train
        and not release_train
        and not args.debug_rollout_only
    )
    pgs = create_placement_groups(args)
    init_tracking(args)

    # The rollout manager determines num_rollout before model initialization.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    try:
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

        # Restore the published version and prefetched batch without advancing either.
        if resume_manifest is not None:
            rollout_data_curr_ref = restore_pipeline_checkpoint(args, actor_model, rollout_manager)
        else:
            actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        # Save only at scheduled policy publications to preserve lag.
        checkpoint_pending = False
        pending_checkpoint = None
        rollout_data_next_future = (
            None if resume_manifest is not None else rollout_manager.generate.remote(args.start_rollout_id)
        )
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if pending_checkpoint is not None and pending_checkpoint.finalize(blocking=False) is not None:
                pending_checkpoint = None
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                rollout_data_next_future = None

            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

            if release_train:
                actor_model.create()

            actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
            if args.use_critic:
                # Reuse actor log-probs in critic targets and actor training.
                logprob_refs = actor_model.async_forward_log_probs(rollout_data_curr_ref)
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref, external_data=logprob_refs)
                if actor_trains:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

            if pending_checkpoint is not None:
                logger.info("Completed training rollout: rollout_id=%s while background checkpoint is pending", rollout_id)

            checkpoint_id = rollout_id
            save_due = release_train or should_run_periodic_action(
                rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
            )
            if pipeline_checkpoints and should_run_periodic_action(
                rollout_id, args.async_checkpoint_interval or args.save_interval, num_rollout_per_epoch, args.num_rollout
            ):
                checkpoint_pending = True
            weights_sync_due = release_train or (rollout_id + 1) % args.update_weights_interval == 0
            pipeline_save_due = checkpoint_pending and weights_sync_due
            if checkpoint_pending and not weights_sync_due:
                logger.info("Deferring full checkpoint at rollout %s to the next scheduled weight sync", rollout_id)
            if save_due and not pipeline_checkpoints:
                # Complete saves before releasing trainers or ending the run.
                force_sync = release_train or rollout_id == args.num_rollout - 1
                if actor_trains:
                    actor_model.save_model(checkpoint_id, force_sync=force_sync)
                if args.use_critic:
                    critic_model.save_model(checkpoint_id, force_sync=force_sync)
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))

            if weights_sync_due:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
                actor_model.update_weights()

            if pipeline_checkpoints:
                ray.get(rollout_manager.acknowledge_checkpoint_batch.remote(rollout_id))

            if pipeline_save_due:
                if pending_checkpoint is not None:
                    pending_checkpoint.finalize(blocking=True)
                    pending_checkpoint = None
                saved_checkpoint = save_pipeline_checkpoint(
                    args, rollout_id, actor_model, critic_model, rollout_manager,
                    has_next_batch=rollout_id + 1 < args.num_rollout,
                    checkpoint_iteration=checkpoint_id,
                )
                if getattr(args, "async_save", False):
                    pending_checkpoint = saved_checkpoint
                checkpoint_pending = False

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))

        if pending_checkpoint is not None:
            pending_checkpoint.finalize(blocking=True)
        if args._keep_actor_resident and actor_model.args.offload_train:
            actor_model.offload()
    except BaseException as training_error:
        try:
            if getattr(args, "rollout_function_path", None) == "slime.rollout.fully_async_rollout.generate_rollout_fully_async":
                ray.get(rollout_manager.shutdown_generation.remote())
            ray.get(rollout_manager.dispose.remote())
        except BaseException as cleanup_error:
            logger.error("Training failed with %s and rollout cleanup failed with %s",
                         type(training_error).__name__, type(cleanup_error).__name__)
            raise training_error from cleanup_error
        raise
    else:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)

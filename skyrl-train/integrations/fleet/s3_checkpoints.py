"""
S3 Checkpoint Uploader for SkyRL Training.

Provides synchronous upload of checkpoints to S3 with local cleanup to prevent disk exhaustion.

Usage:
    from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

    trainer = wrap_trainer_with_s3_upload(trainer, bucket="skyrl-checkpoints", prefix="run-name")

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket name (default: skyrl-checkpoints)
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def upload_checkpoint_to_s3(
    local_dir: str,
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
    delete_local: bool = True,
) -> bool:
    """
    Upload a checkpoint directory to S3 synchronously.

    Args:
        local_dir: Local checkpoint directory path
        bucket: S3 bucket name
        prefix: S3 key prefix
        region: AWS region
        delete_local: If True, delete local checkpoint after successful upload

    Returns:
        True on success, False on failure
    """
    try:
        import boto3
        from botocore.config import Config
        from boto3.s3.transfer import TransferConfig

        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        )

        s3 = boto3.client("s3", region_name=region, config=config)

        local_path = Path(local_dir)
        if not local_path.exists():
            logger.warning(f"Checkpoint directory does not exist: {local_dir}")
            return False

        checkpoint_name = local_path.name
        s3_prefix = f"{prefix}/{checkpoint_name}"

        transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )

        uploaded_files = 0
        total_size = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                file_size = file_path.stat().st_size
                total_size += file_size

                logger.info(f"Uploading {file_path.name} ({file_size / 1e6:.1f} MB) to s3://{bucket}/{s3_key}")

                s3.upload_file(str(file_path), bucket, s3_key, Config=transfer_config)
                uploaded_files += 1

        logger.info(f"Uploaded {checkpoint_name}: {uploaded_files} files, {total_size / 1e9:.2f} GB")

        if delete_local:
            logger.info(f"Deleting local checkpoint: {local_dir}")
            shutil.rmtree(local_dir)

        return True

    except Exception as e:
        logger.error(f"Failed to upload checkpoint {local_dir}: {e}")
        return False


def cleanup_old_local_checkpoints(ckpt_path: str, keep_n: int = 1) -> None:
    """
    Delete old local checkpoints, keeping only the most recent N.

    Args:
        ckpt_path: Base checkpoint directory
        keep_n: Number of recent checkpoints to keep
    """
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        return

    # Find all global_step_* directories
    checkpoint_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda x: int(x.name.split("_")[-1]),
        reverse=True,
    )

    # Delete all but the most recent N
    for old_dir in checkpoint_dirs[keep_n:]:
        logger.info(f"Cleaning up old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            logger.warning(f"Failed to delete {old_dir}: {e}")


def wrap_trainer_with_s3_upload(
    trainer,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    region: Optional[str] = None,
    keep_local: bool = False,
):
    """
    Wrap a SkyRL trainer to upload checkpoints to S3 after each save.

    IMPORTANT: Upload is SYNCHRONOUS (blocking) to ensure disk space is freed
    before the next checkpoint save.

    Args:
        trainer: SkyRL trainer instance
        bucket: S3 bucket (default: from S3_CHECKPOINT_BUCKET env var)
        prefix: S3 prefix (default: from trainer config)
        region: AWS region (default: from AWS_REGION env var or us-east-1)
        keep_local: If True, keep local checkpoints after upload

    Returns:
        The trainer (modified in place)
    """
    bucket = bucket or os.environ.get("S3_CHECKPOINT_BUCKET", "skyrl-checkpoints")
    region = region or os.environ.get("AWS_REGION", "us-east-1")

    # Get prefix from trainer config if not provided
    if prefix is None:
        run_name = getattr(trainer.cfg.trainer, "run_name", None)
        project_name = getattr(trainer.cfg.trainer, "project_name", "skyrl")
        # Include model name in prefix
        model_path = getattr(trainer.cfg.trainer.policy.model, "path", "unknown-model")
        model_name = Path(model_path).name
        prefix = f"{project_name}/{model_name}/{run_name}" if run_name else f"{project_name}/{model_name}"

    # Check if AWS credentials are available
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not aws_key or not aws_secret:
        logger.warning(
            "AWS credentials not found. S3 upload DISABLED. "
            "Falling back to aggressive local cleanup (keeping only 1 checkpoint). "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to enable S3 upload."
        )
        # Still wrap to do aggressive local cleanup
        s3_enabled = False
    else:
        logger.info(f"S3 checkpoint upload ENABLED: s3://{bucket}/{prefix}/")
        s3_enabled = True

    # Store original method
    original_save_checkpoints = trainer.save_checkpoints
    ckpt_path = trainer.cfg.trainer.ckpt_path

    def save_checkpoints_with_s3():
        """Wrapped save_checkpoints that uploads to S3 and cleans up local."""
        # Clean up old local checkpoints BEFORE saving new one to free disk space
        cleanup_old_local_checkpoints(ckpt_path, keep_n=1)

        # Call original save
        original_save_checkpoints()

        global_step = trainer.global_step
        checkpoint_dir = os.path.join(ckpt_path, f"global_step_{global_step}")

        if s3_enabled and os.path.exists(checkpoint_dir):
            # SYNCHRONOUS upload - blocks until complete
            success = upload_checkpoint_to_s3(
                local_dir=checkpoint_dir,
                bucket=bucket,
                prefix=prefix,
                region=region,
                delete_local=not keep_local,
            )
            if not success:
                logger.error(f"S3 upload failed for {checkpoint_dir}, keeping local copy")

    # Monkey-patch the method
    trainer.save_checkpoints = save_checkpoints_with_s3

    return trainer

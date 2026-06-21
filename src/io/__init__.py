"""IO helpers for dataset traversal, stack loading, and metadata."""

from src.io.dataset_loader import (
    IMAGE_EXTENSIONS,
    SIDECAR_EXTENSIONS,
    StackDiscoveryResult,
    discover_stack_folders,
    is_image_file,
    list_image_files,
    list_sidecar_files,
    natural_key,
    natural_path_key,
)
from src.io.metadata import add_metadata_note, build_dataset_metadata
from src.io.stack_builder import (
    StackBuildConfig,
    build_dataset_stacks,
    build_stack_from_folder,
    load_single_image,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "SIDECAR_EXTENSIONS",
    "StackDiscoveryResult",
    "StackBuildConfig",
    "natural_key",
    "natural_path_key",
    "is_image_file",
    "list_image_files",
    "list_sidecar_files",
    "discover_stack_folders",
    "build_dataset_metadata",
    "add_metadata_note",
    "load_single_image",
    "build_stack_from_folder",
    "build_dataset_stacks",
]

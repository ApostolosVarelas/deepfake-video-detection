import logging
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.resnet50 import resnet50_main
from models.swin_transformer import swin_transformer_main
from models.xception import xception_main
from utils.constants import (
    BATCH_SIZE,
    FAKE_DATASET,
    MAIN_OUTPUT_FOLDER,
    MODELS,
    READ_DATASET,
)
from utils.evaluate_models import test_model_faces, test_model_full_frame
from utils.prepare_data import prepare_data_faces, prepare_data_full_frame


def _train_model(
    model_type: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    device: torch.device,
    training_out: str,
    *,
    prep_name: str
) -> Tuple[DataLoader, nn.Module, Optional[str]]:
    """
    Build and train the given model type.

    Args:
        model_type (str): One of 'ResNet50', 'Xception' or
            'SwinTransformer'.
        train_ds (Dataset): Training frames.
        val_ds (Dataset): Validation frames.
        test_ds (Dataset): Test frames.
        device (torch.device): Device to train on.
        training_out (str): Folder for the weights, metrics and plots.
        prep_name (str): Name of the data preparation ('faces' or
            'frames').

    Returns:
        Tuple[DataLoader, nn.Module, Optional[str]]: The test loader,
            the trained model and the best aggregation method.

    Raises:
        ValueError: If the model type is unknown.
    """
    if model_type == "ResNet50":
        test_loader, model, best_method = resnet50_main(
            train_ds, val_ds, test_ds, device, training_out
        )
        return test_loader, model, best_method

    if model_type == "Xception":
        test_loader, model, best_method = xception_main(
            train_ds, val_ds, test_ds, device, training_out
        )
        return test_loader, model, best_method

    if model_type == "SwinTransformer":
        test_loader, model, best_method = swin_transformer_main(
            train_ds, val_ds, test_ds, device, training_out
        )
        return test_loader, model, best_method

    raise ValueError(f"Unknown model type: {model_type}")


def _run_one_test(
    model: nn.Module,
    test_loader: DataLoader,
    test_fn: Callable,
    device: torch.device,
    model_type: str,
    test_out: str,
    best_agg: Optional[str]
) -> None:
    """
    Run one test of a trained model and save its results.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Loader of the test frames.
        test_fn (Callable): test_model_faces or test_model_full_frame.
        device (torch.device): Device to run the model on.
        model_type (str): Name of the model architecture.
        test_out (str): Folder where the results are saved.
        best_agg (Optional[str]): Aggregation method chosen during
            validation.

    Returns:
        None
    """
    os.makedirs(test_out, exist_ok=True)
    test_fn(model, test_loader, device, model_type, test_out, best_agg)


def main() -> None:
    """
    Train every model on face crops and on full frames, and test each
    trained model on both kinds of input.

    Phase 1 trains on face crops and tests on faces and frames. Phase 2
    trains on full frames and tests on frames and faces.

    Returns:
        None
    """
    os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(MAIN_OUTPUT_FOLDER, "DeepFakeDetection.log"),
        filemode="w+",
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---------------------------
    # PHASE 1: PREPARE "faces" ONCE
    # ---------------------------
    logging.info("===== PHASE 1: PREPARING ‘faces’ DATA … =====")
    train_ds_faces, val_ds_faces, test_ds_faces = prepare_data_faces(
        READ_DATASET, FAKE_DATASET
    )

    _, _, test_ds_frames_for_phase1 = prepare_data_full_frame(
        READ_DATASET, FAKE_DATASET
    )

    for model_type in MODELS:
        logging.info(
            f"========== PHASE 1 (FACES) → MODEL: {model_type} =========="
        )

        training_out = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, "faces", "training"
        )
        os.makedirs(training_out, exist_ok=True)

        faces_test_loader, model, best_agg = _train_model(
            model_type,
            train_ds_faces,
            val_ds_faces,
            test_ds_faces,
            device,
            training_out,
            prep_name="faces"
        )

        combo_name = "faces_on_faces"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing"
        )
        _run_one_test(
            model, faces_test_loader, test_model_faces,
            device, model_type, test_out_dir, best_agg
        )

        combo_name = "faces_on_frames"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        frames_test_loader = DataLoader(
            test_ds_frames_for_phase1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_out_dir = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing"
        )
        _run_one_test(
            model, frames_test_loader, test_model_full_frame,
            device, model_type, test_out_dir, best_agg
        )

    # ----------------------------
    # PHASE 2: PREPARE "frames" ONCE
    # ----------------------------
    logging.info("===== PHASE 2: PREPARING ‘frames’ DATA … =====")
    train_ds_frames, val_ds_frames, test_ds_frames = prepare_data_full_frame(
        READ_DATASET, FAKE_DATASET
    )

    faces_test_loader_for_phase2 = DataLoader(
        test_ds_faces,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for model_type in MODELS:
        logging.info(
            f"========== PHASE 2 (FRAMES) → MODEL: {model_type} =========="
        )

        training_out = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, "frames", "training"
        )
        os.makedirs(training_out, exist_ok=True)

        frames_test_loader, model, best_agg = _train_model(
            model_type,
            train_ds_frames,
            val_ds_frames,
            test_ds_frames,
            device,
            training_out,
            prep_name="frames"
        )

        combo_name = "frames_on_faces"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing"
        )
        _run_one_test(
            model, faces_test_loader_for_phase2, test_model_faces,
            device, model_type, test_out_dir, best_agg
        )

        combo_name = "frames_on_frames"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(
            MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing"
        )
        _run_one_test(
            model, frames_test_loader, test_model_full_frame,
            device, model_type, test_out_dir, best_agg
        )


if __name__ == "__main__":
    main()

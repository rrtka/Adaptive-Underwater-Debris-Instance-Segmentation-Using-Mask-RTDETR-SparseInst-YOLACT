"""
Excel Experiment Logger using pandas and openpyxl.

Maintains the exact 8 sheets:
1. Summary
2. Training History
3. Validation History
4. Model Comparison
5. Hardware
6. Checkpoints
7. Qualitative Results
8. VLM Analysis
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional


class ExperimentExcelLogger:
    """Manages persistent experiment logging into a structured multi-sheet Excel file."""

    SHEETS = [
        "Summary",
        "Training History",
        "Validation History",
        "Model Comparison",
        "Hardware",
        "Checkpoints",
        "Qualitative Results",
        "VLM Analysis",
    ]

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self._initialize_file()

    def _initialize_file(self):
        """Creates the Excel workbook with headers if it does not already exist."""
        if not os.path.exists(self.excel_path):
            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                # 1. Summary
                pd.DataFrame({
                    "Key": [
                        "Project Title",
                        "Dataset",
                        "Target Models",
                        "Proposed Model",
                        "Status",
                        "Created At"
                    ],
                    "Value": [
                        "Underwater Marine Debris Detection and Instance Segmentation using Mask RT-DETR with SparseInst IAM and YOLACT Prototype Masks",
                        "SeaClear Underwater Marine Debris",
                        "M1 (Baseline), M2 (+IAM), M3 (+YOLACT Proto), M4 (+IAM +YOLACT Proto)",
                        "M4 (Mask RT-DETR + SparseInst IAM + YOLACT Prototype Branch)",
                        "Initialized",
                        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }).to_excel(writer, sheet_name="Summary", index=False)

                # 2. Training History
                pd.DataFrame(columns=[
                    "Model", "Epoch", "Loss", "Classification Loss", "Box Loss",
                    "Mask Loss", "Learning Rate", "Step Time (s)"
                ]).to_excel(writer, sheet_name="Training History", index=False)

                # 3. Validation History
                pd.DataFrame(columns=[
                    "Model", "Epoch", "Val Loss", "Val Cls Loss", "Val Box Loss", "Val Mask Loss",
                    "Mask mAP", "Box mAP", "AP50", "AP75", "mIoU", "Precision", "Recall",
                    "FPS", "Latency (ms)", "Params (M)", "GFLOPs"
                ]).to_excel(writer, sheet_name="Validation History", index=False)

                # 4. Model Comparison
                pd.DataFrame(columns=[
                    "Model", "IAM", "YOLACT Prototype", "Accuracy", "Mask mAP", "AP50", "AP75",
                    "mIoU", "Precision", "Recall", "FPS", "Params", "GFLOPs"
                ]).to_excel(writer, sheet_name="Model Comparison", index=False)

                # 5. Hardware
                pd.DataFrame(columns=["Property", "Value"]).to_excel(writer, sheet_name="Hardware", index=False)

                # 6. Checkpoints
                pd.DataFrame(columns=[
                    "Model", "Type", "Epoch", "Metric", "Metric Value", "File Path", "Saved At"
                ]).to_excel(writer, sheet_name="Checkpoints", index=False)

                # 7. Qualitative Results
                pd.DataFrame(columns=[
                    "Sample ID", "Image File", "Model", "GT Categories", "Pred Categories",
                    "Pred Scores", "Visualization Path"
                ]).to_excel(writer, sheet_name="Qualitative Results", index=False)

                # 8. VLM Analysis
                pd.DataFrame(columns=[
                    "Sample ID", "Model", "Image Path", "VLM Model",
                    "Identified Objects", "Visibility Challenges",
                    "Plausibility Assessment", "Full Explanation"
                ]).to_excel(writer, sheet_name="VLM Analysis", index=False)

    def log_hardware(self, hardware_info: Dict[str, Any]):
        """Logs detected hardware environment."""
        df_hw = pd.DataFrame([{"Property": k, "Value": str(v)} for k, v in hardware_info.items()])
        self._append_or_replace("Hardware", df_hw, replace=True)

    def log_training_step(self, row: Dict[str, Any]):
        """Appends an epoch row to Training History."""
        df = pd.DataFrame([row])
        self._append_or_replace("Training History", df, replace=False)

    def log_validation_step(self, row: Dict[str, Any]):
        """Appends an evaluation row to Validation History."""
        df = pd.DataFrame([row])
        self._append_or_replace("Validation History", df, replace=False)

    def log_model_comparison(self, comparison_rows: List[Dict[str, Any]]):
        """Updates the Model Comparison sheet."""
        df = pd.DataFrame(comparison_rows)
        self._append_or_replace("Model Comparison", df, replace=True)

    def log_checkpoint(self, row: Dict[str, Any]):
        """Records a saved checkpoint."""
        df = pd.DataFrame([row])
        self._append_or_replace("Checkpoints", df, replace=False)

    def log_qualitative(self, row: Dict[str, Any]):
        """Records qualitative inference output."""
        df = pd.DataFrame([row])
        self._append_or_replace("Qualitative Results", df, replace=False)

    def log_vlm_analysis(self, row: Dict[str, Any]):
        """Records Qwen2.5-VL semantic explanation."""
        df = pd.DataFrame([row])
        self._append_or_replace("VLM Analysis", df, replace=False)

    def update_summary(self, key_value_dict: Dict[str, Any]):
        """Updates project summary metrics and final best model status."""
        current_summary = pd.read_excel(self.excel_path, sheet_name="Summary")
        for k, v in key_value_dict.items():
            if k in current_summary["Key"].values:
                current_summary.loc[current_summary["Key"] == k, "Value"] = str(v)
            else:
                new_row = pd.DataFrame([{"Key": k, "Value": str(v)}])
                current_summary = pd.concat([current_summary, new_row], ignore_index=True)
        self._append_or_replace("Summary", current_summary, replace=True)

    def _append_or_replace(self, sheet_name: str, new_df: pd.DataFrame, replace: bool = False):
        """Helper to append or replace a specific sheet inside the existing workbook."""
        with pd.ExcelWriter(self.excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            if replace:
                # Clear and write
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                try:
                    existing_df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except Exception:
                    combined_df = new_df
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

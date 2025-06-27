import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pathlib

scripts_dir = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import load_data 


class TestDownloadFromKaggle(unittest.TestCase):
    @patch("load_data.os.path.exists")
    @patch("load_data.subprocess.run")
    @patch("load_data.zipfile.ZipFile")
    @patch("load_data.pd.read_csv")
    def test_file_exists_skips_download(self, mock_read_csv, mock_zipfile, mock_subprocess, mock_exists):
        mock_exists.return_value = True
        load_data.download_from_kaggle()
        mock_subprocess.assert_not_called()
        mock_zipfile.assert_not_called()
        mock_read_csv.assert_not_called()

    @patch("load_data.os.path.exists")
    @patch("load_data.subprocess.run")
    @patch("load_data.zipfile.ZipFile")
    @patch("load_data.pd.read_csv")
    @patch("load_data.os.remove")
    def test_download_and_save(self, mock_remove, mock_read_csv, mock_zipfile, mock_subprocess, mock_exists):
        mock_exists.return_value = False

        dummy_df = MagicMock()
        dummy_df.sample.return_value.reset_index.return_value = dummy_df

        dummy_df.to_csv = MagicMock()

        mock_read_csv.return_value = dummy_df
        mock_zipfile.return_value.__enter__.return_value = MagicMock()

        load_data.download_from_kaggle()

        mock_subprocess.assert_called_once_with(
            ["kaggle", "datasets", "download", "-d", load_data.KAGGLE_DATASET], check=True
        )

        zip_path = load_data.KAGGLE_DATASET.split("/")[-1] + ".zip"
        mock_zipfile.assert_called_once_with(zip_path, "r")

        dummy_df.sample.assert_called_once_with(n=5000, random_state=42)
        dummy_df.sample.return_value.reset_index.assert_called_once_with(drop=True)

        dummy_df.to_csv.assert_called_once_with(load_data.OUTPUT_FILE, index=False)

        mock_remove.assert_any_call(zip_path)
        mock_remove.assert_any_call(load_data.FILE_NAME)


if __name__ == "__main__":
    unittest.main()

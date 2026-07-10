"""Test sul rilevamento backend whisper.cpp (CUDA / Vulkan / CPU)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.whisper_cpp import detect_whisper_backend, whisper_decoding_args
import install as install_mod


class DetectWhisperBackendTests(unittest.TestCase):
    def test_cuda_dll_next_to_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "whisper-cli.exe"
            exe.write_bytes(b"x")
            (root / "ggml-cuda.dll").write_bytes(b"x")
            self.assertEqual(detect_whisper_backend(exe), "cuda")

    def test_vulkan_dll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "whisper-cli.exe"
            exe.write_bytes(b"x")
            (root / "ggml-vulkan.dll").write_bytes(b"x")
            self.assertEqual(detect_whisper_backend(exe), "vulkan")

    def test_cpu_when_no_gpu_dll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "whisper-cli.exe"
            exe.write_bytes(b"x")
            (root / "ggml-blas.dll").write_bytes(b"x")
            self.assertEqual(detect_whisper_backend(exe), "cpu")

    def test_cuda_in_lib_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "whisper-cli"
            exe.write_bytes(b"x")
            lib = root / "lib"
            lib.mkdir()
            (lib / "libggml-cuda.so").write_bytes(b"x")
            self.assertEqual(detect_whisper_backend(exe), "cuda")


class InstallGpuDetectTests(unittest.TestCase):
    def test_pick_cublas_asset_preferred(self):
        assets = [
            {"name": "whisper-blas-bin-x64.zip"},
            {"name": "whisper-cublas-12.4.0-bin-x64.zip"},
            {"name": "whisper-bin-x64.zip"},
        ]
        patterns = [
            r"whisper-cublas-12\..*bin-x64\.zip",
            r"whisper-cublas.*bin-x64\.zip",
            r"whisper-blas-bin-x64\.zip",
        ]
        picked = install_mod.pick_asset(assets, patterns)
        self.assertIsNotNone(picked)
        self.assertIn("cublas", picked["name"])

    def test_detect_gpu_nvidia_via_smi(self):
        with mock.patch.object(install_mod, "_nvidia_smi_lists_gpu", return_value=True), \
             mock.patch.object(install_mod, "_windows_wmi_has_nvidia", return_value=False):
            self.assertEqual(install_mod.detect_gpu(), "nvidia")

    def test_detect_gpu_generic_without_nvidia(self):
        with mock.patch.object(install_mod, "_nvidia_smi_lists_gpu", return_value=False), \
             mock.patch.object(install_mod, "_windows_wmi_has_nvidia", return_value=False), \
             mock.patch("platform.system", return_value="Windows"):
            self.assertEqual(install_mod.detect_gpu(), "generic")

    def test_desired_backend_windows(self):
        self.assertEqual(
            install_mod._desired_whisper_backend("Windows", "nvidia"), "cuda")
        self.assertEqual(
            install_mod._desired_whisper_backend("Windows", "generic"), "vulkan")
        self.assertEqual(
            install_mod._desired_whisper_backend("Linux", "generic"), "cpu")

    def test_whisper_backend_helper_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "whisper-cli.exe"
            exe.write_bytes(b"x")
            (root / "ggml-cuda.dll").write_bytes(b"x")
            self.assertEqual(install_mod._whisper_backend(exe), "cuda")
            self.assertTrue(install_mod._whisper_is_cuda(exe))


class WhisperDecodingArgsTests(unittest.TestCase):
    def setUp(self):
        self._saved_temp = os.environ.pop("WHISPER_TEMPERATURE", None)
        self._had_prompt = "WHISPER_INITIAL_PROMPT" in os.environ
        self._saved_prompt = os.environ.pop("WHISPER_INITIAL_PROMPT", None)

    def tearDown(self):
        if self._saved_temp is None:
            os.environ.pop("WHISPER_TEMPERATURE", None)
        else:
            os.environ["WHISPER_TEMPERATURE"] = self._saved_temp
        if self._had_prompt:
            os.environ["WHISPER_INITIAL_PROMPT"] = self._saved_prompt or ""
        else:
            os.environ.pop("WHISPER_INITIAL_PROMPT", None)

    def test_classic_pipeline_has_no_speech_defaults(self):
        args = whisper_decoding_args(language="it", speech_mode=False)
        self.assertNotIn("-mc", args)
        self.assertNotIn("-sow", args)
        self.assertNotIn("-tp", args)
        self.assertNotIn("--prompt", args)

    def test_speech_mode_enables_stutter_preserving_defaults(self):
        args = whisper_decoding_args(language="it", speech_mode=True)
        self.assertEqual(args[args.index("-mc") + 1], "0")
        self.assertIn("-sow", args)
        self.assertEqual(args[args.index("-tp") + 1], "0.8")
        self.assertIn("--prompt", args)
        prompt = args[args.index("--prompt") + 1]
        self.assertIn("ehh", prompt)

    def test_env_temperature_overrides_classic_default(self):
        os.environ["WHISPER_TEMPERATURE"] = "0.6"
        args = whisper_decoding_args(language="it", speech_mode=False)
        self.assertEqual(args[args.index("-tp") + 1], "0.6")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import io
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('downloads', Path(__file__).parents[1] / 'scripts/download_artifacts.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_traversal_and_links_rejected(tmp_path):
    for name in ('../escape', '/absolute', 'C:/escape', '..\\escape'):
        with pytest.raises(ValueError):
            d.safe_path(tmp_path, name)


def test_existing_results_preserved(tmp_path):
    target = tmp_path / 'results.txt'
    d.write_member(io.BytesIO(b'original'), target)
    d.write_member(io.BytesIO(b'original'), target)
    with pytest.raises(RuntimeError):
        d.write_member(io.BytesIO(b'different'), target)
    assert target.read_bytes() == b'original'


def test_selected_teacher_required(tmp_path):
    archive = tmp_path / 'weights.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('other.pth', b'unexpected')
    with pytest.raises(ValueError):
        d.extract(archive, {'format': 'zip', 'member': 'teacher.pth'}, tmp_path / 'out')


def test_zip_restore(tmp_path):
    archive = tmp_path / 'model.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('work/models/model.pt', b'model-data')
    d.extract(archive, {'format': 'zip'}, tmp_path / 'out')
    assert (tmp_path / 'out/work/models/model.pt').read_bytes() == b'model-data'

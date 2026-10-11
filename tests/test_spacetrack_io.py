"""Tests for the local cache logic, using a fake client instead of the real Space-Track API."""

import pandas as pd
from sgp4 import exporter
from sgp4.api import WGS72, Satrec

from geowatch.elements import elements_from_history
from geowatch.spacetrack_io import read_history, update_history


def _record(gp_id: int, jd: float) -> dict:
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        40258,
        jd - 2433281.5,
        0.0,
        0.0,
        0.0,
        0.0002,
        0.1,
        0.001,
        0.3,
        1.0027379 * 2 * 3.141592653589793 / 1440.0,
        0.5,
    )
    l1, l2 = exporter.export_tle(sat)
    epoch = pd.Timestamp("1970-01-01") + pd.to_timedelta((jd - 2440587.5) * 86400, unit="s")
    return {
        "GP_ID": str(gp_id),
        "NORAD_CAT_ID": "40258",
        "OBJECT_NAME": "LUCH (OLYMP-K)",
        "EPOCH": epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "TLE_LINE1": l1,
        "TLE_LINE2": l2,
    }


class FakeSpaceTrack:
    """Mimics SpaceTrackClient.gp_history, including the '>epoch' filter."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.calls: list[dict] = []

    def gp_history(self, **query):
        self.calls.append(query)
        out = self.records
        if "epoch" in query:
            threshold = pd.Timestamp(query["epoch"].lstrip(">"))
            out = [r for r in out if pd.Timestamp(r["EPOCH"]) > threshold]
        return sorted(out, key=lambda r: r["EPOCH"])


def test_first_download_then_incremental_update(tmp_path):
    first_batch = [_record(1, 2461000.25), _record(2, 2461001.25), _record(3, 2461002.25)]
    client = FakeSpaceTrack(first_batch)

    assert update_history(client, 40258, directory=tmp_path) == 3
    assert "epoch" not in client.calls[0]  # first run asks for the whole history

    client.records = first_batch + [_record(4, 2461003.25), _record(5, 2461004.25)]
    assert update_history(client, 40258, directory=tmp_path) == 2
    assert client.calls[1]["epoch"].startswith(">")  # second run only asks for newer sets

    raw = read_history(40258, directory=tmp_path)
    assert list(raw["GP_ID"]) == ["1", "2", "3", "4", "5"]

    assert update_history(client, 40258, directory=tmp_path) == 0  # nothing new


def test_elements_from_history_keeps_latest_publication_of_an_epoch():
    a, b = _record(10, 2461000.25), _record(11, 2461000.25)  # same epoch published twice
    raw = pd.DataFrame([b, a]).astype("string")  # deliberately out of GP_ID order
    el = elements_from_history(raw)
    assert len(el) == 1
    assert el.loc[0, "object_name"] == "LUCH (OLYMP-K)"

from __future__ import annotations

import pytest

from crossformer.data.arec.arec import ArrayRecordBuilder, pack_record


class RecordingDataSource:
    def __init__(self, records):
        self._records = records
        self.batches = []

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]

    def __getitems__(self, indices):
        batch = list(indices)
        self.batches.append(batch)
        return [self._records[i] for i in batch]


def make_builder_with_source(total_records):
    records = [pack_record(i) for i in range(total_records)]
    data_source = RecordingDataSource(records)
    builder = ArrayRecordBuilder(name="test", root="/tmp", version="v1")
    builder._ds = data_source
    builder._meta = {"num_records": total_records}
    return builder, data_source


@pytest.mark.parametrize("total_records", [1, 10, 16_384, 40_010])
def test_iteration_batches_do_not_exceed_chunk_and_yield_all_records(total_records):
    builder, data_source = make_builder_with_source(total_records)

    yielded = list(builder)

    assert yielded == list(range(total_records))
    assert all(len(batch) <= 16_384 for batch in data_source.batches)
    assert sum(len(batch) for batch in data_source.batches) == total_records

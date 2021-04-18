import csv


class Record:
    def __init__(self, warp_id, sm_id, begin, end):
        self.warp_id = warp_id
        self.sm_id = sm_id
        self.start_time = begin
        self.end_time = end

    def duration(self):
        return self.end_time - self.start_time


class Report:
    def __init__(self, records):
        self.records = records

    def get_records_for_sm(self, sm_id):
        return list(filter(lambda record: record.sm_id == sm_id, self.records))

    def get_records(self):
        return self.records


def load_report(filename):
    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            data.append(Record(int(row[0]), int(row[1]), int(row[2]), int(row[3])))

    return Report(data)

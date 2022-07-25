"""Evaluate chunk metrics."""
# the input file has this structure
# token\tgold-pos\tgold-chunk\tpred-chunk
# cut the predicted chunk output from the shallow parse output
# paste it with the gold-pos-chunk file
# if seqeval not installed
# install using pip install seqeval
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from sys import argv


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def process_lines_prepare_gold_and_system_outputs(lines):
    """Process input lines and prepare gold and system outputs."""
    gold_all, pred_all, temp_gold, temp_pred = list(), list(), list(), list()
    for line in lines:
        line = line.strip()
        if line:
            gold, pred = line.split()[-2:]
            temp_gold.append(gold)
            temp_pred.append(pred)
        else:
            assert len(temp_gold) == len(temp_pred)
            gold_all.append(temp_gold)
            pred_all.append(temp_pred)
            temp_gold, temp_pred = list(), list()
    if temp_gold and temp_pred:
        assert len(temp_gold) == len(temp_pred)
        gold_all.append(temp_gold)
        pred_all.append(temp_pred)
    return gold_all, pred_all


def generate_classification_metrics(gold, pred):
    """Generate classification metrics using seqeval package."""
    class_report = ''
    class_report += classification_report(gold, pred, mode='strict', scheme=IOB2) + '\n'
    class_report += 'Accuracy = ' + str(accuracy_score(gold, pred)) + '\n'
    class_report += 'Micro_F1 = ' + str(f1_score(gold, pred))
    return class_report


def write_data_into_file(data, file_path):
    """Write data into a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(data + '\n')


def main():
    """Pass arguments and call functions here."""
    input_file = argv[1]
    output_file = argv[2]
    input_lines = read_lines_from_file(input_file)
    gold_all, pred_all = process_lines_prepare_gold_and_system_outputs(input_lines)
    class_report = generate_classification_metrics(gold_all, pred_all)
    write_data_into_file(class_report, output_file)


if __name__ == '__main__':
    main()

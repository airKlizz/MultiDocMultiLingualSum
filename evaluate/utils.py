from os.path import join


def write_hypotheses(dataset, hypotheses_folder):
    hyp_columns = [
        column for column in dataset.column_names if column[-10:] == "hypothesis"
    ]
    for baseline in hyp_columns:
        with open(join(hypotheses_folder, baseline + ".txt"), "w") as f:
            for hyp in dataset[baseline]:
                f.write("{}\n\n".format(hyp.replace("\n", "")))


def write_references(dataset, hypotheses_folder, summary_colunm_name):
    with open(join(hypotheses_folder, "references.txt"), "w") as f:
        for ref in dataset[summary_colunm_name]:
            f.write("{}\n\n".format(ref.replace("\n", "")))


def write_documents(dataset, hypotheses_folder, document_colunm_name):
    with open(join(hypotheses_folder, "documents.txt"), "w") as f:
        for doc in dataset[document_colunm_name]:
            f.write("{}\n\n".format(doc.replace("\n", "")))


def write_csv(scores, csv_file, rouge_types):
    proper_scores = _get_proper_scores(scores, rouge_types)
    with open(csv_file, "w") as f:
        for line in proper_scores:
            f.write(";".join(line) + "\n")


def write_md(scores, md_file, rouge_types):
    proper_scores = _get_proper_scores(scores, rouge_types)
    with open(md_file, "w") as f:
        f.write("| " + " | ".join(proper_scores[0]) + " |\n")
        f.write(
            "| " + " | ".join(["---" for _ in range(len(proper_scores[0]))]) + " |\n"
        )
        for line in proper_scores[1:]:
            f.write("| " + " | ".join(line) + " |\n")


def _get_proper_scores(scores, rouge_types):
    proper_scores = []
    for baseline, score in scores.items():
        head = []
        proper_score = []
        for rouge_type, measures in rouge_types.items():
            for measure in measures:
                head.append(rouge_type + "." + measure)
                proper_score.append(_get_measure_value(score[rouge_type], measure))
        proper_score.insert(0, baseline)
        proper_scores.append(proper_score)
    head.insert(0, "   ")
    proper_scores.insert(0, head)
    return proper_scores


def _get_measure_value(agg_score, measure):

    value = None
    if "low" in measure:
        value = agg_score.low
    elif "mid" in measure:
        value = agg_score.mid
    elif "high" in measure:
        value = agg_score.high
    else:
        raise ValueError

    if "precision" in measure:
        value = value.precision
    elif "recall" in measure:
        value = value.recall
    elif "fmeasure" in measure:
        value = value.fmeasure
    else:
        raise ValueError

    return f"{value:.2%}"

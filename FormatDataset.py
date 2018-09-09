def read_and_format_data():
    dataset = []
    with open('dataset.txt') as file:
        for line_index, line in enumerate(file):
            for word_index, word in enumerate(line.split()):
                if line_index == 0:
                    dataset.append([])
                if line_index == 0 and word_index == 0:
                    continue
                if word_index == 0:
                    dataset[word_index].append(1 if int(word) >= 0 else 0)
                else:
                    dataset[word_index].append(word)
    return dataset

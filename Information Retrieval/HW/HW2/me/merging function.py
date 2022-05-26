# maybe incorrect
def merging_all(in_dir):
    while True:
        files = os.listdir(in_dir)
        blocks = [file for file in files if file.startwith('dictionary')]
        if len(blocks) == 1:
            return
        if len(blocks) % 2 == 1:
            for block_id in range((len(blocks) - 1) / 2):
                merging_MAO(2 * block_id, 2 * block_id + 1)
            os.rename("dictionary" + str(len(blocks) - 1) + ".txt", "dictionary" + str((len(blocks) - 1) / 2) + ".txt",)
        else:
            for block_id in range(len(blocks) / 2):
                merging_MAO(2 * block_id, 2 * block_id + 1)


# maybe incorrect
def merging_MAO(block1No, block2No):
    dictFile = open("dictionary" + str(block1No) + ".txt", "rb")
    unpickler = pickle.Unpickler(dictFile)
    dictionay_data_1 = unpickler.load()
        
    dictFile = open("dictionary" + str(block2No) + ".txt", "rb")
    unpickler = pickle.Unpickler(dictFile)
    dictionay_data_2 = unpickler.load()
    
    postingFile = open("postings" + str(block1No) + ".txt", "rb")
    unpickler = pickle.Unpickler(postingFile)
    posting_data_1 = unpickler.load()

    postingFile = open("postings" + str(block2No) + ".txt", "rb")
    unpickler = pickle.Unpickler(postingFile)
    posting_data_2 = unpickler.load()

    for term_2 in dictionay_data_2:
        pointer_2 = dictionay_data_2[term_2]["pointer"]
        if term_2 in dictionay_data_1:
            dictionay_data_1[term_2]["documentFrequencies"] += 1
            pointer_1 = dictionay_data_1[term_2]["pointer"]
            for docID_term_2 in posting_data_2[pointer_2]:
                if docID_term_2 not in posting_data_1[pointer_1]:
                    posting_data_1[pointer_1].append(docID_term_2)
        else:
            dictionay_data_1[term_2]["documentFrequencies"] = 1
            posting_data_1.append(posting_data_2[pointer_2])
            dictionay_data_1[term_2]["pointer"] = len(posting_data_1) - 1

    sorted_dict = dict(sorted(dictionay_data_1.items()))

    # write the result of block n and block n+1 to block (n+1) / 2
    # (1, 2) -> 1  (3, 4) -> 2  (5, 6) -> 3  (7, 8) -> 4
    # (1 ~ 8) -> (1 ~ 4)

    dictFile = open("dictionary" + str((block1No + 1) / 2) + ".txt", "wb")
    pickler = pickle.Pickler(dictFile)
    for key in sorted_dict:
        pickler.dump({key: sorted_dict[key]})
    dictFile.close()

    postingFile = open("postings" + str((block1No + 1) / 2) + ".txt", "wb")
    pickler = pickle.Pickler(postingFile)
    for posting in posting_data_1:
        pickler.dump(posting)
    postingFile.close()

    os.remove("dictionary" + str(block2No) + ".txt", "rb")
    os.remove("postings" + str(block2No) + ".txt")
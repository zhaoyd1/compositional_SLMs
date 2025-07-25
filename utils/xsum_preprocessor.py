import pyarrow.parquet as pq
import json
import nltk

def load_parquet(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    # for index, row in df.iterrows():
    #     if len(row["document"]) <= 1:
    #         print(row["document"])
    return df

# for index, row in df.iterrows():
#     print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")
import re
def split_punctuation(text):

    pattern = r"(?<!\w)'|'(?!\w)|[^\w\s'.,]|(?<!\d),|,(?!\d)"
    split_text = re.sub(pattern, lambda match: f" {match.group(0)} ", text)
    split_text = re.sub(r'\s+', ' ', split_text).strip()
    return split_text

def process(doc_line):
    max_sccessive_dot = 0
    cur_sccessive_dot = 0
    for i in range(len(doc_line)):
        if doc_line[i] == ".":
            cur_sccessive_dot += 1
        else:
            if cur_sccessive_dot > max_sccessive_dot:
                max_sccessive_dot = cur_sccessive_dot
            cur_sccessive_dot = 0

    if cur_sccessive_dot > max_sccessive_dot:
        max_sccessive_dot = cur_sccessive_dot

    if max_sccessive_dot > 2:
        sep = max_sccessive_dot * "."
        doc_line_tmp = doc_line.split(sep)
        for i in range(len(doc_line_tmp)):
            if i != len(doc_line_tmp) - 1:
                doc_line_tmp[i] = split_punctuation(doc_line_tmp[i])
            else:
                if len(doc_line_tmp[i]) > 0:
                    flag = False
                    for j in range(len(doc_line_tmp[i]) - 1, -1, -1):
                        if ("A" <= doc_line_tmp[i][j] and doc_line_tmp[i][j] <= "Z") or ("a" <= doc_line_tmp[i][j] and doc_line_tmp[i][j] <= "z") or ("0" <= doc_line_tmp[i][j] and doc_line_tmp[i][j] <= "9"):
                            flag = True
                            break
                    if flag:
                        first_part = doc_line_tmp[i][:j+1]
                        first_part = split_punctuation(first_part)
                        second_part = " ".join(list(doc_line_tmp[i][j+1:]))
                        if second_part != "":
                            doc_line_tmp[i] = first_part + " " + second_part
                        else:
                            doc_line_tmp[i] = first_part
                    else:
                        doc_line_tmp[i] = " ".join(list(doc_line_tmp[i]))
        doc_line = (" " + sep + " ").join(doc_line_tmp)
        doc_line = doc_line.strip()
    else:
        flag = False
        for j in range(len(doc_line) - 1, -1, -1):
            if ("A" <= doc_line[j] and doc_line[j] <= "Z") or ("a" <= doc_line[j] and doc_line[j] <= "z") or ("0" <= doc_line[j] and doc_line[j] <= "9"):
                flag = True
                break
        if flag:
            first_part = doc_line[:j+1]
            first_part = split_punctuation(first_part)
            second_part = " ".join(list(doc_line[j+1:]))
            if second_part != "":
                doc_line = first_part + " " + second_part
            else:
                doc_line = first_part
        else:
            doc_line = " ".join(list(doc_line))

    new_doc_line = ""
    for i in range(len(doc_line)):
        if doc_line[i] == "'" and i != 0 and i != len(doc_line) - 1:
            if (("A" <= doc_line[i-1] and doc_line[i-1] <= "Z") or ("a" <= doc_line[i-1] and doc_line[i-1] <= "z") or ("0" <= doc_line[i-1] and doc_line[i-1] <= "9")) and \
                (("A" <= doc_line[i+1] and doc_line[i+1] <= "Z") or ("a" <= doc_line[i+1] and doc_line[i+1] <= "z") or ("0" <= doc_line[i+1] and doc_line[i+1] <= "9")):
                    new_doc_line += " '"
            else:
                new_doc_line += "'"
        else:
            new_doc_line += doc_line[i]
    doc_line = new_doc_line
    double_ref_count = sum([1 for item in doc_line if item == "\""])
    if double_ref_count > 0:
        if double_ref_count % 2 == 0:
            cnt = 0
            new_doc_line = ""
            for j in range(len(doc_line)):
                if doc_line[j] == "\"":
                    if cnt % 2 == 0:
                        new_doc_line += "``"
                    else:
                        new_doc_line += "''"
                    cnt += 1
                else:
                    new_doc_line += doc_line[j]
            doc_line = new_doc_line
        else:
            if doc_line[-1] == "\"":
                doc_line = doc_line[:-1] + "''"
                new_doc_line = ""
                cnt = 0
                for j in range(len(doc_line)):
                    if doc_line[j] == "\"":
                        if cnt % 2 == 0:
                            new_doc_line += "``"
                        else:
                            new_doc_line += "''"
                        cnt += 1
                    else:
                        new_doc_line += doc_line[j]
                doc_line = new_doc_line
            else:
                cnt = 0
                new_doc_line = ""
                for j in range(len(doc_line)):
                    if doc_line[j] == "\"":
                        if cnt % 2 == 0:
                            new_doc_line += "``"
                        else:
                            new_doc_line += "''"
                        cnt += 1
                    else:
                        new_doc_line += doc_line[j]
                doc_line = new_doc_line

    doc_line = doc_line.replace("(", "-LRB-")
    doc_line = doc_line.replace(")", "-RRB-")
    doc_line = doc_line.replace("[", "-LSB-")
    doc_line = doc_line.replace("]", "-RSB-")
    doc_line = doc_line.replace("{", "-LCB-")
    doc_line = doc_line.replace("}", "-RCB-")  

    return doc_line

def generate_json(df, output_path, tiny=False):
    res = []
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(index)
        document = row["document"].replace("\n", " ")
        document = document.replace("!", ".")
        summary = row["summary"].replace("\n", " ")
        summary = summary.replace("!", ".")
        document = nltk.sent_tokenize(document)
        summary = nltk.sent_tokenize(summary)
        for k in range(len(document)):
            doc_line = document[k]
            doc_line = process(doc_line)
            document[k] = doc_line   
            # print(doc_line)        
            # exit()
        for k in range(len(summary)):
            sum_line = summary[k]
            sum_line = process(sum_line)
            summary[k] = sum_line            
        
        if len(document) > 0 and len(summary) > 0: # avoid empty document and empty summary
            res.append({"document": document, "summary": summary})
        # if len(summary) != 1:
        #     print(summary)
        if tiny:
            if index >= 249:
                break
    # exit()
    with open(output_path, "w") as json_file:
        json.dump(res, json_file, indent=4)


if __name__ == "__main__":
    df = load_parquet('/data/home/zhaoyd/train/0000.parquet')
    generate_json(df, '/data/home/zhaoyd/train/train.json', False)

import pandas as pd
import xml.etree.ElementTree as ET
import os

def parse_fra_file(filepath):
    ext = os.path.splitext(filepath.name)[1].lower() if hasattr(filepath, "name") else os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath, skiprows=0)
        cols = df.columns.str.lower()
        freq_col = [c for c in cols if "freq" in c][0]
        resp_col = [c for c in cols if "resp" in c or "amp" in c or "db" in c][0]
        df = df[[freq_col, resp_col]]
        df.columns = ["frequency", "response"]
        return df

    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath)
        num_cols = df.select_dtypes(include="number").columns[:2]
        df = df[num_cols]
        df.columns = ["frequency", "response"]
        return df

    elif ext == ".xml":
        tree = ET.parse(filepath)
        root = tree.getroot()
        freq, resp = [], []
        for pt in root.findall(".//Point"):
            freq.append(float(pt.find("Frequency").text))
            resp.append(float(pt.find("Response").text))
        return pd.DataFrame({"frequency": freq, "response": resp})

    else:
        raise ValueError(f"Unsupported file format: {ext}")

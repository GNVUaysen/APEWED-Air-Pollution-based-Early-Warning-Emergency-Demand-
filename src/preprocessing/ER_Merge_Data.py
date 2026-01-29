import os
import re
import pandas as pd


INPUT_DIR = r"."

OUTPUT_XLSX = os.path.join(INPUT_DIR, "All_ER_Dataset.xlsx")

def sheet_name_from_filename(filename: str) -> str:

    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(20\d{2})", base)
    if m:
        return m.group(1)  
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", base).strip()
    return cleaned[:31] if cleaned else "Hoja"

def safe_unique_sheet_name(name: str, used: set) -> str:

    original = name[:31]
    name = original
    i = 2
    while name in used:
        suffix = f"_{i}"
        name = (original[:31 - len(suffix)] + suffix) if len(original) + len(suffix) > 31 else (original + suffix)
        i += 1
    used.add(name)
    return name

def main():
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".xlsx") and not f.startswith("~$")
    )

    if not files:
        raise FileNotFoundError(f"Not found .xlsx en: {INPUT_DIR}")

    used_sheet_names = set()

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for f in files:
            path = os.path.join(INPUT_DIR, f)
            sh = safe_unique_sheet_name(sheet_name_from_filename(f), used_sheet_names)


            df = pd.read_excel(path, sheet_name=0)
            

            df = df.iloc[15:].reset_index(drop=True)


            df.to_excel(writer, sheet_name=sh, index=False)

    print(f"Done:\n{OUTPUT_XLSX}")

if __name__ == "__main__":
    main()

import json
import pandas as pd
from pathlib import Path

from main import get_model_id

def read_file(path): 
    with open(path, 'r') as file:
        archive = json.load(file)
        return archive 

if __name__ == "__main__":
    path_to_save =  Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / f"model_{get_model_id()}_x_reports.csv"
    reports_json_path = Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / "patients_reports.json"
    model_findings_json_path = f"../{get_model_id()}_x_ray_findings.json"

    reports_json = read_file(reports_json_path)
    model_findings_json = read_file(model_findings_json_path)

    data = []

    for key in reports_json.keys():
        data.append({
            "id_patient": key,
            "model_response": model_findings_json[key][f"{get_model_id()}-x-ray-findings"]["response"],
            "model_reasoning": model_findings_json[key][f"{get_model_id()}-x-ray-findings"]["reasoning"],
            "real_report": reports_json[key]["real_report"]
        })

    df = pd.DataFrame(data)

    print(df)
    df.to_csv(path_to_save, index=False, sep=",", encoding='utf-8', quoting=1)


    


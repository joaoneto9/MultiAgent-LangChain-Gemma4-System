import json
import pandas as pd
from pathlib import Path

def read_file(path): 
    with open(path, 'r') as file:
        archive = json.load(file)
        return archive 

if __name__ == "__main__":
    path_to_save =  Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / "model_x_reports.csv"
    reports_json_path = Path.home() / "Downloads" / "HUAC-DICOM-TORAX-DCM" / "patients_reports.json"
    gemma4_findings_json_path = "../gemma4_x_ray_findings.json"

    reports_json = read_file(reports_json_path)
    gemma4_findings_json = read_file(gemma4_findings_json_path)

    data = []

    for key in reports_json.keys():
        data.append({
            "id_patient": key,
            "model_response": gemma4_findings_json[key]["Gemma4-x-ray-findings"]["response"],
            "model_reasoning": gemma4_findings_json[key]["Gemma4-x-ray-findings"]["reasoning"],
            "real_report": reports_json[key]['report']
        })

    df = pd.DataFrame(data)

    print(df)
    df.to_csv(path_to_save, index=False, sep=",", encoding='utf-8', quoting=1)


    


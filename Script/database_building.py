import requests
import json

def get_rxcui(drug_name):
    rxcui_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"

    response = requests.get(rxcui_url)
    rxcui_data = response.json()

    # Extract the RxCUI
    rxcui = rxcui_data['idGroup']['rxnormId'][0]
    return rxcui

# Find the drug label
def get_spl_set_id():
    spl_id_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allprops.json?prop=SPL_SET_ID"

    response = requests.get(spl_id_url)
    spl_data = response.json()

    # Extract the first SPL_SET_ID from the list
    spl_set_id = spl_data['propConceptGroup']['propConcept'][0]['propValue']

    print(f"The SPL_SET_ID is: {spl_set_id}")

# dailymed
def dailymed():
    dailymed_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{spl_set_id}.json"

    response = requests.get(dailymed_url)
    label_data = response.json()

    # The 'indications_and_usage' section is often the first element in this list
    indications_text = label_data['indications_and_usage'][0]

    print("--- INDICATIONS AND USAGE ---")
    # Pretty print the JSON for readability
    print(json.dumps(indications_text, indent=2))

if __name__ == "__main__":
    drug_name = "Lisinopril"
    rxcui = get_rxcui(drug_name)
    spl_set_id = get_spl_set_id()
    dailymed()
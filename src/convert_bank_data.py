import pandas as pd
import os

def convert_excel_to_csv():
    input_path = "data/bank.xlsx"
    output_path = "data/bank_efts.csv"
    
    print(f"Reading {input_path}...")
    try:
        # Read the excel file
        df = pd.read_excel(input_path)
        
        # Check if TRANSACTION DETAILS exists
        if "TRANSACTION DETAILS" not in df.columns:
            print("Error: 'TRANSACTION DETAILS' column not found in the excel file.")
            return
            
        # Select and rename the required column
        eft_df = pd.DataFrame()
        eft_df["description"] = df["TRANSACTION DETAILS"]
        
        # Add an eft_id column starting from 1
        eft_df["eft_id"] = range(1, len(eft_df) + 1)
        
        # Reorder columns to match existing system (eft_id, description)
        eft_df = eft_df[["eft_id", "description"]]
        
        # Drop rows with empty descriptions if any
        eft_df = eft_df.dropna(subset=["description"])
        
        # Save to csv
        print(f"Saving to {output_path} with {len(eft_df)} records...")
        eft_df.to_csv(output_path, index=False)
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Make sure you have openpyxl installed (pip install openpyxl)")

if __name__ == "__main__":
    convert_excel_to_csv()

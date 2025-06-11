import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_brain_stroke_dataset(input_filename: str = "../Brain_stroke_prediction_dataset_Raw.csv", output_path: str = "final_dataset.csv") -> pd.DataFrame:
    # 1. Load data
    df = pd.read_csv(input_filename)

    # 2. Hapus missing values
    df = df.dropna()

    # 3. Hapus duplikat
    df = df.drop_duplicates()

    # 4. Binning umur
    bins = [0, 18, 40, 60, 100]
    labels = ['Anak', 'Dewasa Muda', 'Dewasa', 'Lansia']
    df['age'] = pd.cut(df['age'], bins=bins, labels=labels)

    # 5. Label encoding kolom kategorikal
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 6. Standardisasi kolom numerik
    scaler = StandardScaler()
    df[['avg_glucose_level', 'bmi']] = scaler.fit_transform(df[['avg_glucose_level', 'bmi']])

    # 7. Simpan ke file CSV
    df.to_csv(output_path, index=False)

    return df

# Contoh eksekusi langsung jika dijalankan sebagai script
if __name__ == '__main__':
    preprocess_brain_stroke_dataset()


import os
import random
import shutil

MAIN_DATABASE_DIRECTORY = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Baza_Wiedzy_AGH"
EVALUATION_DIRECTORY = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"

TOTAL_BACKGROUND_FILE_COUNT = 170


def sample_balanced_background_files():
    os.makedirs(EVALUATION_DIRECTORY, exist_ok=True)
    existing_files = set(os.listdir(EVALUATION_DIRECTORY))

    files_by_category = {}

    print("Scanning database categories...")
    for category_name in os.listdir(MAIN_DATABASE_DIRECTORY):
        category_path = os.path.join(MAIN_DATABASE_DIRECTORY, category_name)

        if os.path.isdir(category_path):
            files_by_category[category_name] = []
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.md') and file not in existing_files:
                        files_by_category[category_name].append(os.path.join(root, file))

    files_by_category = {k: v for k, v in files_by_category.items() if v}

    if not files_by_category:
        print("No new files to collect.")
        return

    category_count = len(files_by_category)
    files_per_category = TOTAL_BACKGROUND_FILE_COUNT // category_count
    remainder = TOTAL_BACKGROUND_FILE_COUNT % category_count

    print(f"Detected {category_count} main categories. Collecting approximately {files_per_category} files from each.")

    selected_paths = []
    for category, files in files_by_category.items():
        files_to_collect = files_per_category + (1 if remainder > 0 else 0)
        remainder -= 1

        sampled_files = random.sample(files, min(files_to_collect, len(files)))
        selected_paths.extend(sampled_files)
        print(f" -> {category}: selected {len(sampled_files)} files")

    print(f"\nCopying a total of {len(selected_paths)} files...")
    for source_path in selected_paths:
        destination_path = os.path.join(EVALUATION_DIRECTORY, os.path.basename(source_path))
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)


if __name__ == "__main__":
    sample_balanced_background_files()
    print("Balanced background file collection completed.")

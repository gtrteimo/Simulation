import os
import sys

def combine_cuh_files(search_directory=".", output_filename="combined_cuh_content.txt"):
    """
    Finds all .cuh files in a directory (and its subdirectories)
    and concatenates their content into a single text file.

    Args:
        search_directory (str): The directory to search for .cuh files.
                                Defaults to the current directory.
        output_filename (str): The name of the output text file.
                                Defaults to 'combined_cuh_content.txt'.
    """
    cuh_files = []

    # Check if the search directory exists
    if not os.path.isdir(search_directory):
        print(f"Error: Directory '{search_directory}' not found.")
        return

    print(f"Searching for .cuh files in '{search_directory}' and subdirectories...")

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.endswith(".cuh"):
                full_path = os.path.join(root, file)
                cuh_files.append(full_path)

    if not cuh_files:
        print(f"No .cuh files found in '{search_directory}' or its subdirectories.")
        # Remove previous output file if it exists and no new files were found
        if os.path.exists(output_filename):
             print(f"Removing existing output file '{output_filename}' as no files were found.")
             os.remove(output_filename)
        return

    print(f"Found {len(cuh_files)} .cuh files.")
    print(f"Writing combined content to '{output_filename}'...")

    # Write content to the output file
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for cuh_file_path in cuh_files:
                print(f"  Processing: {cuh_file_path}")
                try:
                    with open(cuh_file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()

                    # Add a separator and the file content
                    outfile.write(f"// --- Start of file: {cuh_file_path} ---\n")
                    outfile.write(content)
                    # Ensure a newline at the end of the file content if it doesn't have one
                    if not content.endswith('\n'):
                         outfile.write('\n')
                    outfile.write(f"// --- End of file: {cuh_file_path} ---\n")
                    outfile.write("\n\n") # Add extra newlines between files

                except IOError as e:
                    print(f"Warning: Could not read file '{cuh_file_path}': {e}")
                except Exception as e:
                     print(f"Warning: An unexpected error occurred while processing '{cuh_file_path}': {e}")

        print("Done. Combined content saved to '{output_filename}'.")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_filename}': {e}")


if __name__ == "__main__":
    # Get directory from command line arguments if provided
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = "." # Default to current directory

    combine_cuh_files(target_directory)
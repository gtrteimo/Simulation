import os
import shutil

# --- Configuration ---
# The name of the big file that will be created.
OUTPUT_FILENAME = "combined_output.txt" 


def combine_files_in_folder():
    """
    Finds all files in the current directory, skips the script and output file,
    and concatenates them into one big output file.
    """
    print("Starting file combination process...")
    
    # Get a list of all items (files and directories) in the current folder.
    all_items = os.listdir("./src/cuda")

    # We will collect the paths of the files we actually want to copy here.
    files_to_copy = []
    
    for filename in sorted(all_items): # Sorting makes the order predictable
            
        # If it passed all checks, it's a file we want to copy.
        files_to_copy.append(filename)

    print(f"\nFound {len(files_to_copy)} files to combine.")

    # Open the output file in 'write binary' mode ('wb').
    # This erases the file if it already exists and ensures all file types are handled.
    try:
        with open(OUTPUT_FILENAME, 'wb') as outfile:
            for filename in files_to_copy:
                print(f"  + Adding {filename}...")
                
                # Add a separator to know where one file begins and another ends.
                # We must 'encode' the text separator into bytes to write it in binary mode.
                separator = f"\n\n==================== START OF {filename} ====================\n\n".encode('utf-8')
                outfile.write(separator)
                
                # Open the source file in 'read binary' mode ('rb').
                with open("src/cuda/"+filename, 'rb') as infile:
                    # shutil.copyfileobj is efficient for copying file contents.
                    shutil.copyfileobj(infile, outfile)

            # Add a final separator to mark the end of the combined file.
            final_separator = f"\n\n==================== END OF FILE ====================\n".encode('utf-8')
            outfile.write(final_separator)

        print(f"\nSuccess! All files have been combined into '{OUTPUT_FILENAME}'.")

    except IOError as e:
        print(f"\nAn error occurred: {e}")


# This is the standard entry point for a Python script.
if __name__ == "__main__":
    combine_files_in_folder()
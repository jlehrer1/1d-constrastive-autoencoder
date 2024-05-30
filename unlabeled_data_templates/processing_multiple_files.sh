# Get a list of folder names from the ephys folder

folders=$(ls ./ephys)

#do a for loop 5 times
for ((i=1; i<=12; i++))
do
    # Loop through the folders and get the phy paths
    for folder in $folders
    do
        # Process the data
        python3 processing_script.py --uuid $folder --round $i
    done

    python3 delete_first_file_in_folder.py
    
done




#Take a list of phy paths and download them to the local machine
#This script will download the phy files from the AWS bucket to the local machine
#The script will take a list of phy paths and download them to the local machine

#Import paths from the phy_paths.txt file
while IFS= read -r line
do
    echo "$line"
    #Download the phy files from the AWS bucket to the local machine
    #Create the variable local line
    local_line="${line#braingeneers/}"

    echo "$local_line"

    aws --endpoint https://s3.braingeneers.gi.ucsc.edu s3 cp s3://$line ./$local_line
done < phy_zip_test.txt

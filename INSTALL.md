1-	Follow the instructions of installing [LBCNN](https://github.com/juefeix/lbcnn.torch/blob/master/INSTALL.md).

2- Create a folder for the code:

    > mkdir ./tcs-lbcnn
    
    > cd tcs-lbcnn
    
    
3-	Download the LBCNN repository from [here](https://github.com/juefeix/lbcnn.torch/archive/refs/heads/master.zip).

    > git clone https://github.com/juefeix/lbcnn.torch.git
    
4- Move the contents of 'lbcnn.torch' to the current path. 

    > cp -rf  ./lbcnn.torch/* ./
    
    > rm -rf ./lbcnn.torch
    
5- Download the [TCS-LBCNN](https://github.com/EbrahimAlwajih/TCS-LBCNN/archive/refs/heads/main.zip) repository.

    > git clone https://github.com/EbrahimAlwajih/TCS-LBCNN.git
    
6- Move the contents of 'TCS-LBCNN' to the current path.

    > cp -rf  ./TCS-LBCNN/* ./
    
    > rm -rf ./TCS-LBCNN
    

7- Download the datasets from [here](https://drive.google.com/file/d/1iiw-D4OfcMbASJX10RmlBf-IKU58ArVQ/view?usp=share_link). Unzip the dataset in the same path.

     > curl -L "https://drive.google.com/uc?export=download&confirm=Uq6r&id=1iiw-D4OfcMbASJX10RmlBf-IKU58ArVQ" > Data.tar.gz
     
     > tar -xvzf Data.tar.gz
     
     > rm -rf ./Data.tar.gz
     

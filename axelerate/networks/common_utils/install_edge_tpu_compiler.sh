wget https://packages.cloud.google.com/apt/doc/apt-key.gpg 

sudo apt-key add apt-key.gpg &&

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update && sudo apt-get install -y edgetpu-compiler &&

rm apt-key.gpg

sudo apt-get install -y pciutils cpio &&
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz &&
tar xf l_openvino_toolkit_p_2020.1.023.tgz &&
cd l_openvino_toolkit_p_2020.1.023 && 
sudo -E ./install_openvino_dependencies.sh && 
sed -i 's/decline/accept/g' silent.cfg && 
sudo -E ./install.sh --silent silent.cfg

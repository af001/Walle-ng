#!/bin/bash

yel=$'\e[1;33m'
red=$'\e[1;31m'
end=$'\e[0m'

echo -e "${red}[+] Starting install script${end}"
sleep 2

# Update and install dependencies
echo -e "${red} > Updating and installing dependencies${end}"
apt update && sudo apt -y upgrade && sudo apt -y autoremove
apt install build-essential python3 python3-dev cmake git python3-opencv python3-scipy vim

# Download OpenVINO toolkit
echo -e "${red} > Downloading and installing OpenVINO Toolkit${end}"
wget https://download.01.org/opencv/2019/openvinotoolkit/R2/l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz

# Install OpenVINO Toolkit
mkdir -p /opt/intel/openvino

# Untar OpenVINO Tookit
tar -xf l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz --strip 1 -C /opt/intel/openvino

# Remove tgz
rm -f l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc

# Add USB Rules
echo -e "${red} > Adding USB rules for the Movidius${end}"
usermod -a -G users "$(whoami)"
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh

# Download additional dependencies
echo -e "${red} > Installing additional dependencies${end}"
pip3 install adafruit-blinka adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
# git clone https://github.com/waveshare/Pan-Tilt-HAT

# Modify config.txt in /boot
echo -e "${red} > Modifying /boot/config.txt${end}"
sed -i 's/^dtparam=i2s=on/#dtparam=i2s=on/g' /boot/config.txt
sed -i 's/^dtparam=audio=on/#dtparam=audio=on/g' /boot/config.txt
sed -i '/^\[pi4\]/i # GPIO IRQ\n' /boot/config.txt
sed -i '/^# GPIO IRQ/a dtoverlay=gpio-no-irq' /boot/config.txt

# Reboot
echo
echo -e "${yel}[+] Summary: ${end}"
echo -e "${yel} > Movidus install directory = /opt/intel${end}"
echo -e "${yel} > Walle install directory   = /home/pi/Walle${end}"
echo -e "${yel} > Run walle-ng: cd Walle && python3 walle-ng.py${end}"
echo
echo -e "${red}[+] Done! Rebooting${end}"
sleep 2
reboot

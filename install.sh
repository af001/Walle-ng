#!/bin/bash

yel=$"\e[1;33m"
red=$"\e[1;31m"
end=$"\e[0m"

echo -e "${red}[+] Starting install script${end}"
sleep 2

# Update and install dependencies
echo
echo -e "${red} > Updating and installing dependencies${end}"
sudo apt update && sudo apt -y upgrade && sudo apt -y autoremove
sudo apt -y install build-essential python3 python3-dev cmake git python3-opencv python3-scipy vim

# Download OpenVINO toolkit
echo 
echo -e "${red} > Downloading and installing OpenVINO Toolkit${end}"
wget https://download.01.org/opencv/2019/openvinotoolkit/R2/l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz

# Install OpenVINO Toolkit
sudo mkdir -p /opt/intel/openvino

# Untar OpenVINO Tookit
sudo tar -xf l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz --strip 1 -C /opt/intel/openvino

# Remove tgz
rm -f l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
source ~/.bashrc

# Add USB Rules
echo
echo -e "${red} > Adding USB rules for the Movidius${end}"
sudo usermod -a -G users "$(whoami)"
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh

# Download additional dependencies
echo
echo -e "${red} > Installing additional dependencies${end}"
pip3 install adafruit-blinka adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
# git clone https://github.com/waveshare/Pan-Tilt-HAT

# Modify config.txt in /boot
echo
echo -e "${red} > Modifying /boot/config.txt${end}"
sudo sed -i "s/^dtparam=i2s=on/#dtparam=i2s=on/g" /boot/config.txt
sudo sed -i "s/^dtparam=audio=on/#dtparam=audio=on/g" /boot/config.txt
sudo sed -i "/^\[pi4\]/i # GPIO IRQ\n" /boot/config.txt
sudo sed -i "/^# GPIO IRQ/a dtoverlay=gpio-no-irq" /boot/config.txt

# Ask for headless run, i.e. start on boot
echo
echo -ne "${red}[!] Do you want to run in headless mode? [y/N] ${end}"
read headless

if [[ $headless = "y" || $headless = "Y" ]]; then
        echo -ne "${red} > Enabling headless mode in config.cfg${end}"
	sed -i "s/do_output      =True/do_output      =False/g" config/config.cfg
	echo
	echo -ne "${red}[!] Do you create a startup service for walle-ng? [y/N] ${end}"
	read boot

	if [[ $boot = "y" || $boot = "Y" ]]; then
        	echo -ne "${red} > Enabling walle-ng to run on boot${end}"
        	sudo cp config/walled.service /etc/systemd/system
		sudo systemctl daemon-reload
		sudo systemctl enable walled.service
	fi
fi

# Modify config.cfg with custom AWS settings
echo
echo -ne "${red}[!] Do you want to set up AWS notifications? [y/N] ${end}"
read response

if [[ $response = "y" || $response = "Y" ]]; then
	echo -ne "${red}[+] Enter API Gateway endpoint URL: ${end}"
	read url
	sed -i -e "s|<api_gateway_endpoint_url>|$url|g" config/config.cfg
	echo -ne "${red}[+] Enter API Gateway API key: ${end}"	
	read key
	sed -i -e "s/<api_gateway_api_key>/$key/g" config/config.cfg
fi

# Updating keyboard mapping to us
sudo sed -i 's/XKBLAYOUT="gb"/XKBLAYOUT="us"/g' /etc/default/keyboard
sudo dpkg-reconfigure keyboard-configuration

# Reboot
echo
echo -e "${yel}[+] Summary: ${end}"
echo -e "${yel} > Movidus install directory = /opt/intel${end}"
echo -e "${yel} > Walle install directory   = /home/pi/Walle${end}"
echo -e "${yel} > Run walle-ng: cd Walle && python3 walle-ng.py${end}"
echo
echo -e "${red}[+] Done! Rebooting${end}"
sleep 2
sudo reboot

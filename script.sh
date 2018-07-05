echo "Converting ipynb to py"
jupyter nbconvert *.ipynb --to python
echo "Successfully converted!"
echo "Training the model"
sudo /home/ec2-user/anaconda3/bin/python train.py
echo "Running validation"
sudo /home/ec2-user/anaconda3/bin/python validate.py

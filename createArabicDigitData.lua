-- main.lua

require 'torch'
require 'math'
require 'image'

local matio = require 'matio'
local MHAD = matio.load('/media/eqw/New Volume/lbcnn/Datasets/MHAD/MHADTest.mat')
nData=10000
local data = MHAD.data:type(torch.getdefaulttensortype())
local MHADlabels=MHAD.label
local labels= torch.ByteTensor(nData):fill(0)
for i=1,nData do
--print(testlabels[{1,i}])
--print(bilinguallabel:type())
--print(testlabels:type())
	labels[i]= torch.ByteTensor(1):fill(MHADlabels[{1,i}])+1
end 

print('Type of bilinguallabel: ',labels:type())
print('Type of bilingualdata: ',data:type())

--Create the table to save
Data_to_Write = { data = data, labels = labels }

--Save the table in the /tmp
torch.save("/media/eqw/New Volume/lbcnn/Datasets/MHAD/test_arabic_digit.t7", Data_to_Write)
for i=1, 10 do
--print (table.getn(data))
--print(data[{{i},{1},{},{}}])
--print(data:size())
x=data[{{i},{1},{},{}}]
ou=image.toDisplayTensor{x, padding=2, zoom=4}; image.save(tostring(labels[i])..'.png', ou)
--  image.display(x)

end


-- main.lua

require 'torch'
require 'math'
require 'image'

local matio = require 'matio'
local MHAD = matio.load('/media/eqw/Data/lbcnn/Datasets/cbilingual/en-arTest.mat')
nData=10200
local data = MHAD.data:type(torch.getdefaulttensortype())
local HODAlabels=MHAD.label

--local data=torch.cat(HODAdata,MHADdata,1)
local labels= torch.ByteTensor(nData):fill(0)
for i=1,nData do
	--labels[i]=MNISTlabels[i]
--print('HODAlabels[{1,i}]= ',HODAlabels[{1,i}])
labels[i]= torch.ByteTensor(1):fill(HODAlabels[{1,i}])+1
end 

print('Type of hodalabel: ',labels:type())
print('Type of hodadata: ',data:type())

--Create the table to save
Data_to_Write = { data = data, labels = labels }

--Save the table in the /tmp
 torch.save("/media/eqw/Data/lbcnn/Datasets/cbilingual/test_32x32.t7", Data_to_Write)

 testt7 = torch.load('/media/eqw/Data/lbcnn/Datasets/cbilingual/test_32x32.t7')
 print(testt7)


for i=1, 1 do
--print (table.getn(data))
--print(data[{{i},{1},{},{}}])
--print(data:size())
x=data[{{i},{1},{},{}}]
ou=image.toDisplayTensor{x, padding=2, zoom=4}; image.save(tostring(labels[i])..'.png', ou)
--  image.display(x)

end


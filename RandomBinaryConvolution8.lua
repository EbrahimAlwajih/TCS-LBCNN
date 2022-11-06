-- BinaryConvolution.lua

local THNN = require 'nn.THNN'
local RandomBinaryConvolution8, parent = torch.class('cudnn.RandomBinaryConvolution8', 'cudnn.SpatialConvolution')

function RandomBinaryConvolution8:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end


function RandomBinaryConvolution8:reset()
	local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH):fill(0)
	print('nInputPlane',self.nInputPlane)
	print('nOutputPlane',self.nOutputPlane)
	self.weight[{{},{},{2},{2}}]=-1
        self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW*self.kH)

	local index=torch.Tensor({1,2,3,4,6,7,8,9})
	shuffle (index)
	local i=1
	for nInputPlane = 1,self.nInputPlane do
		shuffle (index) -- for only 8 0randome anchore weights 
		for nOutputPlane = 1,self.nOutputPlane do
			math.randomseed(os.clock())
			local rand1=math.random(1,8)
			self.weight[{{nOutputPlane},{nInputPlane},{index[rand1]}}]=1
			i=i+1
		end
	i=1
	end
        --print(self.weight)
	--print(self.weight(nOutputPlane,nInputPlane,index[i]))
	self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
	--print(self.weight)

	self.bias = nil
	self.gradBias = nil	
	self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0) 	
end

function RandomBinaryConvolution8:accGradParameters(input, gradOutput, scale)
end

function RandomBinaryConvolution8:updateParameters(learningRate)
end

function shuffle (arr)
	size=arr:numel()
	for i=1, size do
		math.randomseed(os.time())
		local rand1=math.random(size)
		arr[i],arr[rand1] = arr[rand1], arr[i]
	end
	return arr
end


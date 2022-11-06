-- main.lua

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'math'
-- require 'SpatialConvolutionPoly'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = nil,nil
local optimState =  nil

if opt.testOnly then
   checkpoint, optimState = checkpoints.best(opt)
   optimState = checkpoint and torch.load(checkpoint.optimFile) or nil
else
   checkpoint, optimState = checkpoints.latest(opt)
   optimState = checkpoint and torch.load(checkpoint.optimFile) or nil
end

-- Create model
local model, criterion = models.setup(opt, checkpoint)
local N = (opt.depth-10)/6
print(model:__tostring__())
print('Number of convolutional layers .. '..#model:findModules('cudnn.SpatialConvolution'))
local x,dx = model:getParameters()
    print('parameters size ..')
    print(#x)
    
--[[
local n_parameters = 0
for i=1, model:size() do
   local params = model:get(i):parameters()
   if params then
     local weights = params[1]
     local biases  = params[2]
     n_parameters  = n_parameters + weights:nElement() + biases:nElement()
   end
end
local params, gradParams=model:getParameters()
local n_par= params:size(1)
print (' Number of n-par= ')
	print(n_par)
print (' Number of Parameters= ')
print(n_parameters)

--]]

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local checkpoint, optimState = checkpoints.best(opt)
   local optimState = checkpoint and torch.load(checkpoint.optimFile) or nil

   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
    --opt.LR = epoch <= 30 and 1e-3 or epoch <= 50 and 1e-4 or 1e-5

   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end
   checkpoints.save(opt, epoch, model, trainer.optimState, bestModel)

   -- update logger
   accLogger:add{['% train accuracy'] = trainTop1, ['% test accuracy'] = testTop1}
   errLogger:add{['% train error']    = trainLoss, ['% test error']    = testLoss}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   accLogger:plot()
   errLogger:plot()
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
local function countParameters(model)
local n_parameters = 0
for i=1, model:size() do
   local params = model:get(i):parameters()
   if params then
     local weights = params[1]
     local biases  = params[2]
     n_parameters  = n_parameters + weights:nElement() + biases:nElement()
   end
end
return n_parameters
end

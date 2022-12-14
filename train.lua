-- train.lua

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model:cuda()
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
   trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)


   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local output = self.model:forward(self.input):float()
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5
      lossSum = lossSum + loss
      N = N + 1

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f LR %1.5f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5, self.optimState.learningRate))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1Sum/N,
      ['% top5 accuracy (train set)'] = top5Sum/N,
      ['avg loss (train set)'] = lossSum/N
   }
   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local lossSum = 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      lossSum = lossSum + loss
      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5
      N = N + 1

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f) LR %1.5f'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N, self.opt.LR))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   testLogger:add{
      ['% top1 accuracy (test set)'] = top1Sum/N,
      ['% top5 accuracy (test set)'] = top5Sum/N,
      ['avg loss (test set)'] = lossSum/N
   }
   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end



function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.oldLR2==1 then
      decay = epoch >= 51 and 2 or epoch >= 31 and 1 or 0
   elseif self.opt.oldLR2==2 then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   else
--elseif self.opt.dataset == 'cifar10' then
      -- decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      decay = epoch >= 61 and 2 or epoch >= 31 and 1 or 0
      --decay = epoch >= 101 and 2 or epoch >= 91 and 1.5 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer

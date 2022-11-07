-- opts.lua

local M = { }
require 'lfs'
local curr_dir = lfs.currentdir()

function M.parse(arg)
   local home = os.getenv("HOME")
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------

   --cmd:option('-data',  home .. '/Datasets', 'path to dataset')
   cmd:option('-data',  './data', 'path to dataset')	
   cmd:option('-dataset',    'mnist', 'Options: imagenet | cifar10 | svhn | frgc | cifar10 | mhad | dbilingual '  )
   cmd:option('-manualSeed', 3,          'Manually set RNG seed')
   cmd:option('-GPU',        1,          'Default GPu to use')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')   
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')   
   cmd:option('-save',       '../cache/' , 'Path to save')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2, 'number of data loading threads')
   cmd:option('-subset',          false, 'use subset or not' )
   cmd:option('-trsize',          2000, 'number of train data')
   cmd:option('-tstsize',         1000, 'number of test data')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         80,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       20,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   cmd:option('-resume',          '../cache/',  'Path to directory containing checkpoint')
   ---------- Optimization options ----------------------
---------- Optimization options ----------------------
   cmd:option('-LR',              1e-3,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-3,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet-dense-felix', 'Options: resnet-dense-felix | resnet-tcsbinary-felix8 | resnet-tcsbinary-felix | resnet-binary-felix | resnet-binary | resnet-binary-felix8 | resnet-dense | resnet-csbinary-felix8 | resnet-csbinary-felix ')
   cmd:option('-depth',        15,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'true',  'Share gradInput tensors to reduce memory usage, better than optnet')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         10,     'Number of classes in the dataset')
   cmd:option('-stride',           1,      'Striding for Convolution, equivalent to pooling')
   cmd:option('-sparsity',         0.5,    'Percentage of sparsity in pre-defined LB filters') 
   cmd:option('-nInputPlane',      3,      'number of input channels')   -- RGB channels for color images
   cmd:option('-numChannels',      128,    'number of intermediate channels') -- output channels
   cmd:option('-full',             512,    'number of hidden units in FC') -- 
   cmd:option('-numWeights',       384,    'number of fixed binary weights') -- Anchor weights == LBC filters
   cmd:option('-convSize',           3,    'LB convolutional filter size')
-------------Temp--------------
   cmd:option('-oldLR2',           0,    'to run the experiments with LR Decay: 2 to run the main LR used with LBCNN | 1 to run first LR2 (30,20,30) | 0 to run new LR2 (30,30,20)')
   cmd:text()

   local opt = cmd:parse(arg or {})
   kSparsity = opt.sparsity
   opt.save = paths.concat(opt.save, opt.dataset .. '_' .. tostring(opt.netType) .. '_' .. tostring(opt.depth) .. '_' .. tostring(opt.numChannels) .. '_' .. tostring(opt.numWeights) .. '_' .. tostring(opt.full) .. '_' .. tostring(opt.sparsity) .. '_' .. tostring(opt.convSize) .. '_' .. tostring(opt.batchSize))
   opt.resume = opt.save
   opt.optimState = opt.save
   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'

   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 1000      
   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 10
      opt.nInputPlane = 3
      opt.view = 6*6
   elseif opt.dataset == 'svhn' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 10
      opt.nInputPlane = 3
      opt.view = 6*6
   elseif opt.dataset == 'mnist' then
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 10
      opt.nInputPlane = 1
      opt.view = 6*6
   elseif opt.dataset == 'mhad' then

      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 10
      opt.nInputPlane = 1
      opt.view = 6*6
	elseif opt.dataset == 'dbilingual' then

      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 16
      opt.nInputPlane = 1
      opt.view = 6*6
   elseif opt.dataset == 'frgc' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 1000 or opt.nEpochs
      opt.nClasses = 466
      opt.nInputPlane = 1
      opt.view =6*6
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M

require 'nn'
require 'cunn'

local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

restrick_en=true;  
--network='large' --large, small_layer_20
network='small_layer_20' --large, small_layer_20

----------------------------------------------------------
if restrick_en then
  print('============================')
  print('=== restrick_en enable ! ===')
  print('============================')
else 
  print('============================')
  print('=== restrick_en disable !===')
  print('============================')
end

local ResNet = nn.Sequential()

local function ConvBNReLU_restrick(nInputPlane, nOutputPlane, downsample, restrick_en)
  local resblock = nn.Sequential()
  local concat   = nn.ConcatTable()
  local vgg2     = nn.Sequential()

  if downsample then  --henry: dotted line of the resnet architecture
    vgg2:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 2,2, 1,1))
    vgg2:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    vgg2:add(backend.ReLU(true))
    vgg2:add(backend.SpatialConvolution(nOutputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    vgg2:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    concat:add(vgg2)
    if restrick_en then
      --[[resnet paper:  
                  When the dimensions increase (dotted line shortcuts
                  in Fig. 3), we consider two options: (A) The shortcut still
                  performs identity mapping, with extra zero entries padded
                  for increasing dimensions. This option introduces no extra
                  parameter; (B) The projection shortcut in Eqn.(2) is used to
                  match dimensions (done by 1x1 convolutions). For both
                  options, when the shortcuts go across feature maps of two
                  sizes, they are performed with a stride of 2.
      --]]
      local IdentyDownSample = backend.SpatialConvolution(nInputPlane, nOutputPlane, 1,1, 2,2, 0,0)
      concat:add(IdentyDownSample)
      print('=== restrick_en enable +1 ===')
    end
  else
    vgg2:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    vgg2:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    vgg2:add(backend.ReLU(true))
    vgg2:add(backend.SpatialConvolution(nOutputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    vgg2:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    concat:add(vgg2)
    if restrick_en then
      concat:add(nn.Identity())
      print('=== restrick_en enable +1 ===')
    end
  end

  resblock:add(concat)
  resblock:add(nn.CAddTable())
  resblock:add(backend.ReLU(true))
  return resblock
end

local function ConvBNReLU_restrick_group(nInputPlane, nOutputPlane, num, downsample, restrick_en)
  local group = nn.Sequential()
  group:add(ConvBNReLU_restrick(nInputPlane, nOutputPlane, downsample, restrick_en))
  for i = 1, num-1 do
    group:add(ConvBNReLU_restrick(nOutputPlane, nOutputPlane, false, restrick_en))
  end
  return group
end

if network == 'large' then
  --follow paper setting: imagenet
  local n64  = 3
  local n128 = 4
  local n256 = 6
  local n512 = 3
  ResNet:add(backend.SpatialConvolution(3, 64, 3,3, 2,2, 1,1))  --OUT: 16X16
  ResNet:add(nn.SpatialBatchNormalization(16, 1e-3))
  ResNet:add(backend.ReLU(true))
  ResNet:add(ConvBNReLU_restrick_group(64,  64, n64 , false, restrick_en)) --OUT: 16X16
  ResNet:add(ConvBNReLU_restrick_group(64, 128, n128, true , restrick_en)) --OUT:  8X8
  ResNet:add(ConvBNReLU_restrick_group(128,256, n256, true , restrick_en)) --OUT:  4X4
  ResNet:add(ConvBNReLU_restrick_group(256,512, n512, true , restrick_en)) --OUT:  2X2
  
  classifier2 = nn.Sequential()
  --classifier2:add(nn.Dropout(0.5))  
  classifier2:add(backend.SpatialMaxPooling(2,2,1,1,0,0))
  classifier2:add(nn.Reshape(512))
  classifier2:add(nn.Linear(512,10))
  classifier2:add(nn.LogSoftMax())
  ResNet:add(classifier2)

elseif network == 'small_layer_20' then
  --follow paper setting: cifar10
  local n16 = 3
  local n32 = 3
  local n64 = 3
  ResNet:add(backend.SpatialConvolution(3, 16, 3,3, 1,1, 1,1))  --OUT: 32X32
  ResNet:add(nn.SpatialBatchNormalization(16, 1e-3))
  ResNet:add(backend.ReLU(true))
  ResNet:add(ConvBNReLU_restrick_group(16,  16, n16 , false, restrick_en)) --OUT: 32X32
  ResNet:add(ConvBNReLU_restrick_group(16,  32, n32 , true , restrick_en)) --OUT: 16X16
  ResNet:add(ConvBNReLU_restrick_group(32,  64, n64 , true , restrick_en)) --OUT:  8X8
  
  classifier2 = nn.Sequential()
  --classifier2:add(nn.Dropout(0.5))  
  classifier2:add(backend.SpatialAveragePooling(8,8,1,1,0,0))
  classifier2:add(nn.Reshape(64))
  classifier2:add(nn.Linear(64,10))
  classifier2:add(nn.LogSoftMax())
  ResNet:add(classifier2)

else
    print('Unknown model type')
    error()
end
-------------------------------------

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(ResNet)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return ResNet


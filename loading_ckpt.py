def initialize_model_from_cfg(args,gpu_id=0):
    model = model_builder.Generalized_RCNN()
    model.eval()
    
    if args.cuda():
        model.cuda()
    
    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info('loading checkpoint %s:',load_name)
        checkpoint = torch.load(load_name,map_location=lambda storage,loc:storage)
        load_ckpt(model,checkpoint['model'])
        
    model = nn.DataParallel(model,cpu_keywords=['im_info','roidb'],minibatch=True)
    
    return model
    
    
def load_ckpt(model,ckpt):
    mapping,_ = model.detectron_weight_mapping
    state_dict = {}
    for name in ckpt:
        if mapping[name]:
            state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict,strict=False)
    
    
    
def save_ckpt(output_dir,args,model,optimizer):
    if args.no_save:
        return
        
    ckpt_dir = os.path.join(output_dir,'ckpt')
    if not os.path.exists:
        os.makedirs(ckpt_dir)
    
    save_name = os.path.join(ckpt_dir,'model_{}_{}.pth'.format(args.epoch,args.step))
    if isinstance(model,nn.Dataparallel):
        model = model.module
        
    torch.save({
        'epoch':args.epoch,
        'step':args.step,
        'iters_per_epoch':args.iters_per_epoch,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict()
                },save_name)
                
    logger.info('save_model:%s',save_name)

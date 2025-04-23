import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Dict, Any

class Runner:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        assign_target_fn,
        device,
        scheduler=None,
        assign_target_kwargs: Dict[str, Any] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        use_tal: bool = False
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.scheduler = scheduler
        self.assign_target_fn = assign_target_fn
        self.assign_target_kwargs = assign_target_kwargs or {}
        self.use_tal = use_tal
        
        self.batch_loss = []
        self.epoch_loss = []
        self.val_loss = []
        
    def _run_train_epoch(self, dataloader, verbose=True):
        self.model.train()
        loss_values = []
        
        for images, targets in (pbar := tqdm(dataloader, desc=f"Process train epoch", leave=False)):
            images = images.to(self.device)
            outputs = self.model(images)
            
            anchors = self.model.anchors.view(-1, 4)
            accum_loss = 0.0
            
            for i in range(images.shape[0]):
                gt_boxes = targets['boxes'][i].to(self.device)
                gt_labels = targets['labels'][i].to(self.device)
                
                # Prepare prediction data for TAL if needed
                pred_data = {}
                if self.use_tal:
                    pred_boxes = self.model.decode_bboxes(outputs[0][i])
                    pred_cls = torch.softmax(outputs[2][i], dim=-1)
                    pred_data = {
                        'pred_boxes': pred_boxes,
                        'pred_cls': pred_cls
                    }
                
                # Assign targets using either TAL or regular assignment
                target_offsets, target_objectness, target_cls = self.assign_target_fn(
                    anchors, gt_boxes, gt_labels, self.model.num_classes,
                    **pred_data,
                    **self.assign_target_kwargs
                )
                
                # Calculate loss
                loss = self.criterion(
                    (outputs[0][i], outputs[1][i], outputs[2][i]),
                    (target_offsets, target_objectness, target_cls)
                )
                accum_loss += loss
                
            # Backward pass
            self.optimizer.zero_grad()
            accum_loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            loss_values.append(accum_loss.item())
            if verbose:
                pbar.set_postfix({'loss': f'{accum_loss.item():.4f}'})
                
        return loss_values
    
    def _run_val_epoch(self, dataloader, verbose=True):
        self.model.eval()
        loss_values = []
        
        with torch.no_grad():
            for images, targets in (pbar := tqdm(dataloader, desc=f"Process val epoch", leave=False)):
                images = images.to(self.device)
                outputs = self.model(images)
                
                anchors = self.model.anchors.view(-1, 4)
                accum_loss = 0.0
                
                for i in range(images.shape[0]):
                    gt_boxes = targets['boxes'][i].to(self.device)
                    gt_labels = targets['labels'][i].to(self.device)
                    
                    # Prepare prediction data for TAL if needed
                    pred_data = {}
                    if self.use_tal:
                        pred_boxes = self.model.decode_bboxes(outputs[0][i])
                        pred_cls = torch.softmax(outputs[2][i], dim=-1)
                        pred_data = {
                            'pred_boxes': pred_boxes,
                            'pred_cls': pred_cls
                        }
                    
                    # Assign targets using either TAL or regular assignment
                    target_offsets, target_objectness, target_cls = self.assign_target_fn(
                        anchors, gt_boxes, gt_labels, self.model.num_classes,
                        **pred_data,
                        **self.assign_target_kwargs
                    )
                    
                    # Calculate loss
                    loss = self.criterion(
                        (outputs[0][i], outputs[1][i], outputs[2][i]),
                        (target_offsets, target_objectness, target_cls)
                    )
                    accum_loss += loss
                    
                loss_values.append(accum_loss.item())
                if verbose:
                    pbar.set_postfix({'loss': f'{accum_loss.item():.4f}'})
                    
        return loss_values
    
    def train(self, num_epochs, verbose=True):
        for epoch in (epoch_pbar := tqdm(range(1, num_epochs+1), desc="Train epoch", total=num_epochs)):
            # Train
            loss = self._run_train_epoch(self.train_dataloader, verbose=verbose)
            self.batch_loss.extend(loss)
            self.epoch_loss.append(np.mean(self.batch_loss[-len(self.train_dataloader):]))
            
            # Validate if validation dataloader is provided
            if self.val_dataloader is not None:
                val_loss = self._run_val_epoch(self.val_dataloader, verbose=verbose)
                self.val_loss.append(np.mean(val_loss))
                
            if verbose:
                epoch_pbar.set_postfix({
                    'train_loss': f'{self.epoch_loss[-1]:.4f}',
                    'val_loss': f'{self.val_loss[-1]:.4f}' if self.val_dataloader is not None else 'N/A'
                }) 
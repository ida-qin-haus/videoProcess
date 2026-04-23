  New flag: --iphone (bundles all three improvements)                                                                   
  python transcribe.py recording.mov --iphone                                                                           
  This activates:                                                                                                       
  1. iPhone-specific prompt — instructs Claude to identify and rejoin soft-wrapped lines, and ignore UI chrome          
  2. Auto-crop — probes the first frame, crops 10% from top (status bar + address bar) and 10% from bottom (tab bar +   
  safe area), scaling automatically to any iPhone resolution                                                            
  3. Blur filtering — skips momentum-blurred frames using Laplacian variance                                            
                                                                                                                        
  Fine-grained overrides (work standalone or to override --iphone defaults):                                            
  # Override crop values (e.g. your phone has a bigger address bar)                                                     
  python transcribe.py recording.mov --iphone --crop-top 300 --crop-bottom 150
                                                                                                                        
  # Blur filtering without the iPhone preset                                                                            
  python transcribe.py recording.mov --skip-blurry --blur-threshold 80                                                  
                                                                                                                        
  # Manual crop on a non-iPhone recording                                                                               
  python transcribe.py recording.mov --crop-top 50 --crop-bottom 80      

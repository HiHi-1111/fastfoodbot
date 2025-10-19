"""
Script to fix all bot issues from steps2.md
Run this to apply all fixes at once
"""
import os
import shutil

def fix_template_paths():
    """Fix template paths in fast_food_bot.py"""
    file_path = 'fast_food_bot.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Update _load_phase2_templates base path
    content = content.replace(
        "            base = 'images'",
        "            base = os.path.join('images', 'phase 2', 'Fries Types')"
    )
    
    # Fix 2: Update template keys
    content = content.replace(
        "                'fries': 'fries.png',",
        "                'fries_classic': 'fries.png',"
    )
    content = content.replace(
        "                'thick_fries': 'thick_fries.png',",
        "                'fries_thick': 'thick_fries.png',"
    )
    
    # Fix 3: Update default return value
    content = content.replace(
        "            return 'fries', best_score",
        "            return 'fries_classic', best_score"
    )
    
    # Fix 4: Update target_type check
    content = content.replace(
        "        target_type = type_name if type_name in ('fries', 'thick_fries', 'onion_rings') else 'fries'",
        "        target_type = type_name if type_name in ('fries_classic', 'fries_thick', 'onion_rings') else 'fries_classic'"
    )
    
    # Fix 5: Update phase2_size_templates return
    content = content.replace(
        "        self._phase2_size_templates = templates if templates else None",
        "        self._phase2_size_templates = templates if templates else {}"
    )
    
    # Fix 6: Update best_name default
    content = content.replace(
        "        best_name, best_score = 'fries', 0.0",
        "        best_name, best_score = 'fries_classic', 0.0"
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Fixed fast_food_bot.py template paths")

def add_click_throttle():
    """Add click throttling to ingredient phase"""
    file_path = 'fast_food_bot.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update click delay from 0.4 to 0.085 for ingredients
    content = content.replace(
        "                        self.click_n_times(item, count)\n                        time.sleep(0.4)",
        "                        self.click_n_times(item, count)\n                        time.sleep(0.085)"
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Added click throttling for ingredients")

def update_button_mappings():
    """Update button mappings for fries types"""
    file_path = 'bot_params.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add mappings for new fries types
    content = content.replace(
        '    "fries": [0.871, 0.403],',
        '    "fries": [0.871, 0.403],\n    "fries_classic": [0.871, 0.403],'
    )
    content = content.replace(
        '    "thick_fries": [0.871, 0.486],',
        '    "thick_fries": [0.871, 0.486],\n    "fries_thick": [0.871, 0.486],'
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Updated button mappings")

def main():
    print("Applying fixes from steps2.md...\n")
    
    # Backup files
    for file in ['fast_food_bot.py', 'bot_params.json']:
        if os.path.exists(file):
            shutil.copy(file, f"{file}.backup")
            print(f"[OK] Backed up {file}")
    
    print()
    
    # Apply fixes
    fix_template_paths()
    add_click_throttle()
    update_button_mappings()
    
    print("\n[OK] All fixes applied successfully!")
    print("\nBackup files created with .backup extension")
    print("Run 'python fast_food_bot.py' to test the bot")

if __name__ == "__main__":
    main()

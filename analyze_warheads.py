import sys
import json
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit module not found. Please install it: pip install rdkit-pypi", file=sys.stderr)
    sys.exit(1)

ALL_REACTION_PATTERNS = {
    'michael_addition': ['[C,c]=[C,c]-[C,c,S,s]=[O]', '[C,c]=[C,c]S(=O)=O', '[C]=[C][S](=O)=O', '[C,c]=[C,c]C#N', 'O=C1C=CC(=O)N1', 'C#CC(=O)', '[C,c]=[C,c][N+](=O)[O-]', '[C,c]=[C,c]c1ncccc1', '[C]=[C][C]#[N]', '[C;H1,H2]=[C;H1]C(=O)[N,n,O,S]', '[C;H1,H2]=[C;H1]S(=O)(=O)[#6]', '[C;H1,H2]=[C;H1]c1ncccc1', '[C;H1,H2]=[C;H1]C#N', '[c,C]#[C]C(=O)[N,n,O,S]', '[C]#[C]-[C]=[O]', '[c,C]=[c,C][c]'],
    'nucleophilic_addition': ['[C,c;H1](=[O;X1])', '[C;X3](=[O;X1])([#6])([#6])', '[C;H1;X3](=O)', '[C;X3](=O)O[#6]', '[C;X3](=O)Cl', '[C;X3](=O)[OX2H1]', '[C;X3]=[N;X3]', '[CX2]#N', '[CX3]=[CX3]-C(=O)[#6]', 'C#C-C(=O)', 'O1CC1', 'N=C=O', '[NX3]=C([OX1])'],
    'nucleophilic_substitution': ['[F,Cl,Br,I]CC(=O)', '[*][F,Cl,Br,I]', 'ClCC(=O)N', '[#6]S(=O)(=O)F', 'C(=O)NS(=O)(=O)', 'S(=O)(=O)[F,Cl,Br,I]', '[C,c](=O)[F,Cl,Br,I]', '[C;X4]1[N;X3][C;X4]1', '[#6]C(=O)O[#6]', '[#6]S(=O)(=O)O[#6]', '[#7]C(=O)O[#6]', '[C;X4]1[O;X2][C;X4]1', 'S(=O)(=O)O[C,c]', '[C,c](=[O,S])[F,Cl,Br,I]', '[c,C]S(=O)(=O)[F,Cl,Br,I]', '[CH2]([F,Cl,Br,I])C(=O)', '[#7]-c1nscn1', '[#7]c1nscn1', '[#7]-c(a):[n,s]'],
    'carbamoylation': ['[#7]C(=O)O[#6]', '[#7]C(=O)O[#6]C(F)(F)F', '[#7]C(=O)OC(C(F)(F)F)C(F)(F)F', '[#7]C(=O)O[#6]([#6])(F)', '[#7]C(=O)Oc1ccc([N+](=O)[O-])cc1', '[#7]C(=O)[F,Cl,Br,I]', '[#7]=[C]=[O]', '[#7]C(=O)OC(F)', '[#7]C(=O)ON'],
    'phosphorylation': ['[#8][P](=O)([#8])[#8]', '[#6][P](=O)([#8])[#8]'],
    'disulfide_formation': ['[#7]C(=S)SSC(=S)[#7]', '[SH]', 'SSc1ncccc1', '[#7]C(=S)[#7]', '[#16]-[#16]'],
    'nt_dithiocarbamate': ['[#7]C(=S)S[#6]'],
    'ring-opening': ['[#16;R][#7;R]([#6]=O)'],
    'transesterification': ['[#7]C(=O)OC(C(F)(F)F)C(F)(F)F', '[#6]C(=O)O[#6]', '[#6]C(=O)S[#6]'],
    'phosphonate_addition': ['[P](=O)(O)[F,Cl,Br,I]', '[P](=O)(O)O'],
    'epoxide_opening': ['[C;X4]1[O;X2][C;X4]1', '[C;X4]1[N;X3][C;X4]1'],
    'boronic_acid': ['B(O)O'],
    'beta_lactam': ['[N;X3]1[C;X4][C;X4][C;X3](=O)1'],
    'gamma_lactam_opening': ['[C;X3](=O)[N;X3]1[C;X4][C;X4][C;X4]1']
}

def find_all_reaction_types(mol):
    if not mol:
        return {}
    
    results = {}
    
    for reaction_type, patterns in ALL_REACTION_PATTERNS.items():
        found_patterns = []
        
        for p_str in patterns:
            try:
                patt = Chem.MolFromSmarts(p_str)
                if patt is None:
                    continue
                if mol.HasSubstructMatch(patt):
                    found_patterns.append(p_str)
            except Exception as e:
                pass
        
        if found_patterns:
            results[reaction_type] = list(set(found_patterns))
    
    return results

def get_receptor_atom_name(resi_name):
    resi_name_upper = str(resi_name).upper().strip()
    
    if '-' in resi_name_upper:
        resi_name_upper = resi_name_upper.split('-')[0].strip()
    
    atom_map = {
        'CYS': 'SG',
        'SER': 'OG',
        'THR': 'OG1',
        'LYS': 'NZ',
        'TYR': 'OH',
        'NT': 'N-terminal'
    }
    
    if resi_name_upper in atom_map:
        return atom_map[resi_name_upper]
    else:
        return None

def main_interactive_loop():
    
    output_filename = "warhead_analysis_results.json"
    
    print("=" * 80)
    print("=== 欢迎使用交互式弹头分析工具 (自动检测所有反应类型) ===")
    print("=" * 80)
    print("本工具将自动遍历所有已知反应类型并检测匹配的弹头模式。")
    print("请输入 'quit' 退出程序。\n")

    while True:
        smiles_input = input("\n[1/2] 请输入 SMILES 字符串: \n> ")
        if smiles_input.lower() == 'quit':
            break
            
        residue_input = input(f"\n[2/2] 请输入目标残基名称 (例如 CYS, SER, 或留空跳过): \n> ")
        if residue_input.lower() == 'quit':
            break

        print("\n" + "=" * 80)
        print("正在分析...")
        print("=" * 80)

        receptor_atom = None
        if residue_input.strip():
            receptor_atom = get_receptor_atom_name(residue_input)
        
        mol = Chem.MolFromSmiles(smiles_input)
        detected_reactions = {}
        
        if not mol:
            print("  - [失败] RDKit 无法解析此 SMILES 字符串。")
        else:
            mol = Chem.AddHs(mol)
            mol.SetProp("_Name", "InteractiveInput")
            detected_reactions = find_all_reaction_types(mol)

        results = {
            "Input_SMILES": smiles_input,
            "Input_Residue": residue_input if residue_input.strip() else "N/A",
            "Inferred_Receptor_Atom": receptor_atom,
            "Detected_Reaction_Types": detected_reactions,
            "Total_Reaction_Types_Found": len(detected_reactions)
        }

        print("\n" + "=" * 80)
        print("分析结果")
        print("=" * 80)
        
        if residue_input.strip():
            if receptor_atom:
                print(f"\n[受体信息]")
                print(f"  残基名称: {residue_input.upper()}")
                print(f"  推断原子: {receptor_atom}")
            else:
                print(f"\n[受体信息]")
                print(f"  残基名称: {residue_input.upper()}")
                print(f"  警告: 无法推断唯一的反应原子")
        
        print(f"\n[配体分析]")
        print(f"  SMILES: {smiles_input}")
        
        if not mol:
            print(f"  状态: ✗ SMILES 无法解析")
        elif not detected_reactions:
            print(f"  状态: ✓ SMILES有效，但未检测到已知反应类型的弹头")
            print(f"  说明: 该分子可能不含共价弹头，或弹头类型不在检测范围内")
        else:
            print(f"  状态: ✓ 检测到 {len(detected_reactions)} 种反应类型")
            print(f"\n[检测到的反应类型及匹配的SMARTS模式]")
            for idx, (reaction_type, patterns) in enumerate(detected_reactions.items(), 1):
                reaction_name = reaction_type.replace('_', ' ').title()
                print(f"\n  {idx}. {reaction_name}")
                print(f"     匹配数量: {len(patterns)} 个SMARTS模式")
                for i, pattern in enumerate(patterns, 1):
                    print(f"       {i}) {pattern}")

        print("\n" + "=" * 80)
        try:
            with open(output_filename, 'a', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                f.write("\n" + "-" * 80 + "\n")
            print(f"✓ 结果已追加到 {output_filename}")
        except Exception as e:
            print(f"✗ 保存结果到文件失败: {e}", file=sys.stderr)
        
        print("=" * 80)

    print("\n程序已退出。感谢使用！")

if __name__ == "__main__":
    main_interactive_loop()


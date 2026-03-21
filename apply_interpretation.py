#!/usr/bin/env python3
"""
ADMET 예측값에 Empirical Threshold를 적용하여 Interpretation을 생성하는 스크립트
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def parse_threshold(threshold_str: str) -> Dict:
    """Empirical Threshold 문자열을 파싱하여 규칙 딕셔너리 반환"""
    threshold_str = str(threshold_str).strip()
    
    if pd.isna(threshold_str) or threshold_str == '':
        return {'type': 'molecular_property'}
    
    # 물음표 범위 (0.04?20, 1?3 형태) - 물음표는 범위 구분자
    if '?' in threshold_str:
        try:
            # 단위 제거
            cleaned = threshold_str.replace('L/kg', '').replace('ml/min/kg', '').replace('log mol/L', '').strip()
            parts = cleaned.split('?')
            if len(parts) == 2:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return {
                    'type': 'range_check',
                    'min': min_val,
                    'max': max_val,
                    'label_if_in_range': 'Proper range',
                    'label_if_out_range': 'Out of range'
                }
        except:
            pass
    
    # 범위 기반 (0-30 / 30-70 / 70-100 형태 또는 0-5 / 5-15 / >15 형태)
    # 단위 먼저 제거 (ml/min/kg 형태의 / 때문에 split 전에 제거 필요)
    unit_removed = threshold_str.replace('ml/min/kg', '').replace('log cm/s', '').replace('log mol/L', '').replace('%', '').strip()
    
    if '/' in unit_removed:
        parts = [p.strip() for p in unit_removed.split('/')]
        if len(parts) == 3:
            try:
                ranges = []
                for part in parts:
                    part_cleaned = part.strip()
                    
                    if '-' in part_cleaned:
                        start, end = part_cleaned.split('-')
                        ranges.append((float(start.strip()), float(end.strip())))
                    elif part_cleaned.startswith('>'):
                        value = float(part_cleaned[1:].strip())
                        ranges.append((value, float('inf')))
                    elif part_cleaned.startswith('<'):
                        value = float(part_cleaned[1:].strip())
                        ranges.append((float('-inf'), value))
                    else:
                        # 숫자만 있는 경우
                        try:
                            value = float(part_cleaned)
                            ranges.append((value, float('inf')))
                        except:
                            pass
                
                if len(ranges) == 3:
                    return {
                        'type': 'three_range',
                        'ranges': ranges,
                        'labels': ['excellent', 'medium', 'poor']
                    }
            except Exception as e:
                pass
    
    # 단일 비교 (> -5.15 log cm/s 형태)
    if threshold_str.startswith('>'):
        try:
            # 단위 제거 후 숫자 추출
            cleaned = threshold_str[1:].replace('log cm/s', '').replace('log mol/L', '').replace('%', '').strip()
            value = float(cleaned)
            return {'type': 'greater_than', 'value': value, 'label_if_true': 'excellent', 'label_if_false': 'poor'}
        except:
            pass
    elif threshold_str.startswith('<'):
        try:
            # 단위 제거 후 숫자 추출
            cleaned = threshold_str[1:].replace('%', '').replace('log cm/s', '').strip()
            value = float(cleaned)
            return {'type': 'less_than', 'value': value, 'label_if_true': 'excellent', 'label_if_false': 'poor'}
        except:
            pass
    
    # 복합 범위 (-4 to 0.5 log mol/L 형태)
    if 'to' in threshold_str.lower():
        try:
            cleaned = threshold_str.lower().replace('log mol/L', '').strip()
            parts = cleaned.split('to')
            if len(parts) == 2:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return {
                    'type': 'range_check',
                    'min': min_val,
                    'max': max_val,
                    'label_if_in_range': 'Proper range',
                    'label_if_out_range': 'Out of range'
                }
        except:
            pass
    
    # 복합 조건 (Gas > 4 / Solid < 8 형태) - 분자 속성으로 처리
    if 'Gas' in threshold_str or 'Solid' in threshold_str or 'liquid' in threshold_str.lower():
        return {'type': 'molecular_property'}
    
    # 분자 속성 (threshold 적용 없음)
    if 'Molecular property' in threshold_str:
        return {'type': 'molecular_property'}
    
    return {'type': 'molecular_property'}


def apply_interpretation(value: float, threshold_rule: Dict, column_name: str) -> str:
    """값과 threshold 규칙을 기반으로 interpretation 적용"""
    if pd.isna(value):
        return 'N/A'
    
    rule_type = threshold_rule.get('type', 'molecular_property')
    
    if rule_type == 'molecular_property':
        return 'Molecular property'
    
    elif rule_type == 'three_range':
        ranges = threshold_rule['ranges']
        labels = threshold_rule['labels']
        
        # 범위 체크: ranges[0] = (min0, max0), ranges[1] = (min1, max1), ranges[2] = (min2, inf)
        if ranges[0][0] <= value <= ranges[0][1]:
            return labels[0]  # excellent
        elif ranges[1][0] <= value <= ranges[1][1]:
            return labels[1]  # medium
        elif value > ranges[2][0]:  # 마지막 범위는 > 기준
            return labels[2]  # poor
        else:
            # 범위 밖인 경우 (예: 음수)
            return labels[0] if value < ranges[0][0] else labels[2]
    
    elif rule_type == 'greater_than':
        if value > threshold_rule['value']:
            return threshold_rule['label_if_true']
        else:
            return threshold_rule['label_if_false']
    
    elif rule_type == 'less_than':
        if value < threshold_rule['value']:
            return threshold_rule['label_if_true']
        else:
            return threshold_rule['label_if_false']
    
    elif rule_type == 'range_check':
        min_val = threshold_rule['min']
        max_val = threshold_rule['max']
        if min_val <= value <= max_val:
            return threshold_rule['label_if_in_range']
        else:
            return threshold_rule['label_if_out_range']
    
    return 'Unknown'


def process_interpretations():
    """ADMET Guideline과 예측 결과를 기반으로 Interpretation 적용"""
    
    # 파일 읽기 (인코딩 처리)
    print("Reading ADMET_Guideline.csv...")
    try:
        guideline_df = pd.read_csv('/home/doyamoon/admet/mga_inference/ADMET_Guideline.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            guideline_df = pd.read_csv('/home/doyamoon/admet/mga_inference/ADMET_Guideline.csv', encoding='cp949')
        except:
            guideline_df = pd.read_csv('/home/doyamoon/admet/mga_inference/ADMET_Guideline.csv', encoding='latin-1')
    
    print("Reading output_compound.csv...")
    try:
        output_df = pd.read_csv('/home/doyamoon/admet/mga_inference/output_compound.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            output_df = pd.read_csv('/home/doyamoon/admet/mga_inference/output_compound.csv', encoding='cp949')
        except:
            output_df = pd.read_csv('/home/doyamoon/admet/mga_inference/output_compound.csv', encoding='latin-1')
    
    # Guideline을 딕셔너리로 변환
    guideline_dict = {}
    for _, row in guideline_df.iterrows():
        col_name = row['Column Name']
        guideline_dict[col_name] = {
            'category': row['ADMET Category'],
            'description': row['Description'],
            'threshold': row['Empirical Threshold'],
            'interpretation': row['Interpretation']
        }
    
    # 결과 테이블 생성
    results = []
    
    for idx, row in output_df.iterrows():
        smiles = row['SMILES']
        result_row = {'SMILES': smiles}
        
        # 각 컬럼에 대해 처리
        for col in output_df.columns:
            if col == 'SMILES':
                continue
            
            value = row[col]
            result_row[f'{col}_value'] = value
            
            # Guideline에서 해당 컬럼 찾기
            if col in guideline_dict:
                guideline = guideline_dict[col]
                threshold_str = guideline['threshold']
                threshold_rule = parse_threshold(threshold_str)
                interpretation = apply_interpretation(value, threshold_rule, col)
                
                result_row[f'{col}_interpretation'] = interpretation
                result_row[f'{col}_category'] = guideline['category']
                result_row[f'{col}_description'] = guideline['description']
            else:
                result_row[f'{col}_interpretation'] = 'N/A'
                result_row[f'{col}_category'] = 'Unknown'
                result_row[f'{col}_description'] = 'Unknown'
        
        results.append(result_row)
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame(results)
    
    # CSV로 저장
    output_path = '/home/doyamoon/admet/mga_inference/interpreted_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Interpreted results saved to {output_path}")
    
    # 요약 테이블 생성 (SMILES, Property, Value, Interpretation만 포함)
    summary_rows = []
    for idx, row in output_df.iterrows():
        smiles = row['SMILES']
        for col in output_df.columns:
            if col == 'SMILES':
                continue
            
            value = row[col]
            if col in guideline_dict:
                guideline = guideline_dict[col]
                threshold_str = guideline['threshold']
                threshold_rule = parse_threshold(threshold_str)
                interpretation = apply_interpretation(value, threshold_rule, col)
                
                summary_rows.append({
                    'SMILES': smiles,
                    'Property': col,
                    'Category': guideline['category'],
                    'Description': guideline['description'],
                    'Predicted Value': value,
                    'Empirical Threshold': threshold_str,
                    'Interpretation': interpretation
                })
    
    summary_df = pd.DataFrame(summary_rows)
    # 중복 제거 (같은 SMILES-Property 쌍은 한 번만)
    summary_df = summary_df.drop_duplicates(subset=['SMILES', 'Property'])
    summary_path = '/home/doyamoon/admet/mga_inference/interpreted_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}")
    
    return results_df, summary_df


def generate_report(summary_df: pd.DataFrame):
    """결과를 요약한 문서 생성"""
    
    report_lines = []
    # 중복 제거
    summary_df_unique = summary_df.drop_duplicates(subset=['SMILES', 'Property'])
    
    report_lines.append("# ADMET 예측 결과 Interpretation 보고서\n")
    report_lines.append(f"생성 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"총 화합물 수: {summary_df_unique['SMILES'].nunique()}\n")
    report_lines.append(f"총 예측 속성 수: {summary_df_unique['Property'].nunique()}\n\n")
    
    # 카테고리별 요약 (중복 제거 후)
    report_lines.append("## 카테고리별 Interpretation 요약\n")
    category_summary = summary_df_unique.groupby(['Category', 'Interpretation']).size().reset_index(name='Count')
    for category in sorted(category_summary['Category'].unique()):
        report_lines.append(f"### {category}\n")
        cat_data = category_summary[category_summary['Category'] == category]
        for _, row in cat_data.iterrows():
            report_lines.append(f"- {row['Interpretation']}: {row['Count']}건\n")
        report_lines.append("\n")
    
    # 화합물별 상세 결과 (중복 제거)
    report_lines.append("## 화합물별 상세 결과\n")
    for smiles in summary_df_unique['SMILES'].unique():
        report_lines.append(f"### SMILES: {smiles}\n")
        smiles_data = summary_df_unique[summary_df_unique['SMILES'] == smiles]
        
        # 카테고리별로 그룹화
        for category in sorted(smiles_data['Category'].unique()):
            report_lines.append(f"#### {category}\n")
            cat_data = smiles_data[smiles_data['Category'] == category]
            for _, row in cat_data.iterrows():
                report_lines.append(
                    f"- **{row['Description']}** ({row['Property']}): "
                    f"예측값 = {row['Predicted Value']:.4f}, "
                    f"Interpretation = {row['Interpretation']}\n"
                )
            report_lines.append("\n")
    
    # Interpretation별 통계 (중복 제거 후)
    report_lines.append("## Interpretation별 통계\n")
    interpretation_stats = summary_df_unique['Interpretation'].value_counts()
    for interpretation, count in interpretation_stats.items():
        percentage = (count / len(summary_df_unique)) * 100
        report_lines.append(f"- {interpretation}: {count}건 ({percentage:.2f}%)\n")
    
    report_content = ''.join(report_lines)
    
    # 파일로 저장
    report_path = '/home/doyamoon/admet/mga_inference/interpretation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report saved to {report_path}")
    
    return report_content


if __name__ == '__main__':
    print("Starting interpretation process...")
    results_df, summary_df = process_interpretations()
    
    print("Generating report...")
    generate_report(summary_df)
    
    print("\nProcess completed successfully!")
    print(f"- Detailed results: interpreted_results.csv")
    print(f"- Summary table: interpreted_summary.csv")
    print(f"- Report: interpretation_report.md")

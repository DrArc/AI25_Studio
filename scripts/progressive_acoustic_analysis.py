# Progressive Acoustic Analysis Module
# This module provides 40% sampling with continue functionality for acoustic analysis

class ProgressiveAcousticAnalysis:
    def __init__(self, main_window):
        self.main_window = main_window
        self.progressive_analysis_state = None
    
    def initialize_progressive_analysis(self, ifc_file_path):
        """Initialize progressive analysis state"""
        try:
            import ifcopenshell
            ifc_file = ifcopenshell.open(ifc_file_path)
            all_spaces = ifc_file.by_type("IfcSpace")
            
            self.progressive_analysis_state = {
                'current_batch': 0,
                'total_spaces': len(all_spaces),
                'analyzed_spaces': set(),
                'all_spaces': all_spaces,
                'batch_size': max(1, int(len(all_spaces) * 0.4)),  # 40% of total
                'ifc_file_path': ifc_file_path
            }
            
            return True
        except Exception as e:
            print(f"Error initializing progressive analysis: {e}")
            return False
    
    def get_current_batch_spaces(self):
        """Get spaces for the current batch"""
        if not self.progressive_analysis_state:
            return []
        
        state = self.progressive_analysis_state
        current_batch = state['current_batch']
        batch_size = state['batch_size']
        all_spaces = state['all_spaces']
        
        start_idx = current_batch * batch_size
        end_idx = min(start_idx + batch_size, len(all_spaces))
        
        # Get spaces for current batch (excluding already analyzed ones)
        available_spaces = []
        for i in range(start_idx, end_idx):
            space = all_spaces[i]
            if space.GlobalId not in state['analyzed_spaces']:
                available_spaces.append(space)
        
        return available_spaces
    
    def run_batch_analysis(self):
        """Run acoustic analysis for the current batch"""
        try:
            if not self.progressive_analysis_state:
                return "Progressive analysis not initialized"
            
            state = self.progressive_analysis_state
            current_batch = state['current_batch']
            batch_size = state['batch_size']
            analyzed_spaces = state['analyzed_spaces']
            
            # Get spaces for current batch
            available_spaces = self.get_current_batch_spaces()
            
            if not available_spaces:
                return "No more spaces to analyze in this batch"
            
            # Run analysis on current batch
            import ifcopenshell
            ifc_file = ifcopenshell.open(state['ifc_file_path'])
            analysis_results = []
            total_analyzed = 0
            total_failures = 0
            total_passes = 0
            
            analysis_results.append(f"🔍 Acoustic Analysis (Batch {current_batch + 1}):")
            analysis_results.append("=" * 40)
            
            for i, space in enumerate(available_spaces):
                try:
                    space_info = self.main_window.analyze_single_space(space, ifc_file)
                    if space_info:
                        total_analyzed += 1
                        failures = self.main_window.check_acoustic_failures(space_info)
                        severity = self.main_window.calculate_acoustic_severity(failures)
                        
                        # Mark as analyzed
                        analyzed_spaces.add(space.GlobalId)
                        
                        if severity in ["high", "critical"]:
                            total_failures += 1
                            analysis_results.append(f"   ❌ {space_info['name']} (ID: {space_info['global_id']}) [Severity: {severity}]")
                            for failure in failures:
                                analysis_results.append(f"      - {failure}")
                        else:
                            total_passes += 1
                            analysis_results.append(f"   ✅ {space_info['name']} (ID: {space_info['global_id']}) [Severity: {severity}]")
                except Exception as e:
                    print(f"Error analyzing space {i}: {e}")
                    continue
            
            # Summary for this batch
            analysis_results.append(f"\n📈 Batch {current_batch + 1} Summary:")
            analysis_results.append(f"• Spaces analyzed in this batch: {total_analyzed}")
            analysis_results.append(f"• Passing spaces: {total_passes}")
            analysis_results.append(f"• Failing spaces: {total_failures}")
            if total_analyzed > 0:
                analysis_results.append(f"• Failure rate: {(total_failures/total_analyzed*100):.1f}%")
            
            # Overall progress
            total_analyzed_overall = len(analyzed_spaces)
            analysis_results.append(f"\n📊 Overall Progress:")
            analysis_results.append(f"• Total spaces analyzed: {total_analyzed_overall}/{state['total_spaces']}")
            analysis_results.append(f"• Progress: {(total_analyzed_overall/state['total_spaces']*100):.1f}%")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"Error in progressive acoustic analysis: {e}"
    
    def continue_to_next_batch(self):
        """Continue to the next batch"""
        if self.progressive_analysis_state:
            self.progressive_analysis_state['current_batch'] += 1
            return True
        return False
    
    def finish_analysis(self):
        """Finish the progressive analysis"""
        if self.progressive_analysis_state:
            del self.progressive_analysis_state
            return True
        return False
    
    def reset_analysis(self):
        """Reset the progressive analysis state"""
        if self.progressive_analysis_state:
            del self.progressive_analysis_state
            return True
        return False
    
    def get_progress_info(self):
        """Get current progress information"""
        if not self.progressive_analysis_state:
            return None
        
        state = self.progressive_analysis_state
        total_analyzed = len(state['analyzed_spaces'])
        total_spaces = state['total_spaces']
        current_batch = state['current_batch']
        batch_size = state['batch_size']
        remaining_spaces = total_spaces - total_analyzed
        
        return {
            'total_spaces': total_spaces,
            'analyzed_spaces': total_analyzed,
            'current_batch': current_batch,
            'batch_size': batch_size,
            'remaining_spaces': remaining_spaces,
            'progress_percentage': (total_analyzed / total_spaces * 100) if total_spaces > 0 else 0
        } 
import React from 'react';

export interface PluginParams {
  [key: string]: any;
}

export interface FormComponentProps {
  params: PluginParams;
  onChange: (params: PluginParams) => void;
}

export interface IndicatorUIPlugin {
  /** The unique identifier matching the backend plugin name (e.g., 'bbands_signal') */
  name: string;
  /** Human-readable label for the UI dropdown */
  label: string;
  /** Default parameters when the plugin is first selected */
  defaultParams: PluginParams;
  /** The React component responsible for rendering the parameter inputs */
  FormComponent: React.FC<FormComponentProps>;
}

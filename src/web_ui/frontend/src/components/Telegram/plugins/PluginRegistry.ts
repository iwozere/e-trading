import React from 'react';
import { IndicatorUIPlugin } from './types';
import BollingerBandsForm, { defaultBBandsParams } from './BollingerBandsForm';

class PluginRegistry {
  private plugins: Map<string, IndicatorUIPlugin> = new Map();

  constructor() {
    // Register default plugins here
    this.register({
      name: 'bbands_signal',
      label: 'Bollinger Bands',
      defaultParams: defaultBBandsParams,
      FormComponent: BollingerBandsForm
    });
  }

  register(plugin: IndicatorUIPlugin) {
    this.plugins.set(plugin.name, plugin);
  }

  getPlugin(name: string): IndicatorUIPlugin | undefined {
    return this.plugins.get(name);
  }

  getAllPlugins(): IndicatorUIPlugin[] {
    return Array.from(this.plugins.values());
  }
}

export const pluginRegistry = new PluginRegistry();

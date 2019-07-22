import sys
import math

class DynamicConsoleTable(object):
    
    def __init__(self, layout):
        self.layout = layout
        self.header = '|'
        self.divider = '|'
        self.heavy_divider = '|'
        for i in range(len(layout)):
            space = max(0, layout[i]['width'] - len(layout[i]['name']))
            header_string = ' ' * int(space / 2.0) + layout[i]['name'] + ' ' * int(math.ceil(space / 2.0))
            self.header += ' ' + header_string + ' |'
            self.divider += '-' * (len(header_string) + 2) + '|'
            self.heavy_divider += '=' * (len(header_string) + 2) + '|'
        self.updated = False
    
    def _format_arg(self, arg, properties):
        if arg == '':
            return ' ' * max(len(properties['name']), properties['width'])
        s = str(arg)
        fixes = len(properties.get('prefix', '')) + len(properties.get('suffix', ''))
        if properties.get('format', False):
            s = properties['format'].format(arg)
        elif isinstance(arg, float):
            s = ('{:.' + str(max(0, properties['width'] - (len(str(int(arg))) + 1 + fixes))) + 'f}').format(arg)
        if properties.get('prefix', False):
            s = properties['prefix'] + s
        if properties.get('suffix', False):
            s += properties['suffix']
        s = s[:properties['width']]
        space = (max(len(properties['name']), properties['width']) - len(s))
        if properties.get('align', None) == 'left':
            return s + ' ' * space
        elif properties.get('align', None) == 'center':
            return ' ' * int(space / 2.0) + s + ' ' * int(math.ceil(space / 2.0))
        else:
            return ' ' * space + s
    
    def print_header(self, heavy=True):
        if self.updated:
            self.finalize()
        print self.divider if not heavy else self.heavy_divider
        print self.header
        print self.divider if not heavy else self.heavy_divider
        
    def print_divider(self, heavy=False):
        if self.updated:
            self.finalize()
        print self.divider if not heavy else self.heavy_divider
    
    def finalize(self, heavy=False, divider=True):
        if not self.updated:
            self.update()
        self.updated = False
        print ('\n' + (self.divider if not heavy else self.heavy_divider)) if divider else ''
    
    def update(self, *args):
        self.updated = True
        args = list(args)
        while len(args) < len(self.layout):
            args.append('')
        row = '|'
        for i in range(len(self.layout)):
            formatted = self._format_arg(args[i], self.layout[i])
            row += ' ' + formatted + ' |'
        sys.stdout.write('\r' + row)
        sys.stdout.flush()

export const renderBytesNumber = (value: number): [number, string] => {
    const suffixes = ['', 'kB', 'MB', 'GB', 'TB', 'PB'],
        { min, log, round, pow } = Math,
        power = min(0 | (log(value) / log(1024)), suffixes.length - 1),
        base = round(value / pow(1024, power)),
        suffix = suffixes[power];

    return [base, suffix];
};
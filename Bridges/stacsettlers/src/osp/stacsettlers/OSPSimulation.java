package osp.stacsettlers;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import representation.FVGeneratorFactory;
import soc.disableDebug.D;
import soc.robot.FactoryDescr;
import soc.robot.stac.simulation.Simulation;
import soc.server.SOCServer;
import soc.server.logger.SOCLogger;

/**
 * StacSettlers' headless games (its Simulation, unchanged) with Open Seat Agents among the robots: a config's agent line
 * of the kind "osp" (Agent=COUNT,NAME,osp) seats robots whose decisions an Open Seat Agent makes (OSPRobotFactory), next
 * to StacSettlers' own kinds (stac, mcts, originalSS, ...), and the games run on OSPServer, through which those robots
 * read the true game. The config is parsed by Simulation's own code (reached by reflection, since it is private), with
 * the osp lines taken out and their robots put back in their place among the agents; the results go where
 * Simulation's go (results/, logs_server/ in the working directory).
 *
 * Usage: java -cp build:STACSETTLERS_JAR -Dosp.agent="COMMAND" [-Dosp.log=DIR] [-Dosp.debug=true]
 *        osp.stacsettlers.OSPSimulation CONFIG
 * e.g. a config for 10 games of one Open Seat Agent against three Stac agents:
 *        Games=10 / Log=false / ChatNegotiation=true / ForceEndTurns=false / ~ / Hexfarer_vs_Stac / Trades=true /
 *        Agent=1,Hexfarer,osp / Agent=3,Stac,stac,TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5 (one per line)
 */
public final class OSPSimulation
{
    private OSPSimulation() {}

    public static void main(String[] args) throws Exception
    {
        if (args.length != 1)
        {
            System.err.println("Usage: OSPSimulation CONFIG");
            System.exit(2);
        }
        FVGeneratorFactory.initialise();
        List<String> lines = Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8);
        // The config's games, for the stop rule's default
        for (String line : lines)
            if (line.startsWith("Games="))
                FailureLimit.games = Integer.parseInt(line.substring("Games=".length()).trim());
        Simulation simulation = parse(lines);
        simulation.start();
        // StacSettlers leaves its threads to end by themselves; the games are over and written
        System.exit(0);
    }

    /** A simulation of the config, as Simulation.parseConfig sets one up, with the osp robots and on OSPServer */
    @SuppressWarnings("unchecked")
    static Simulation parse(List<String> lines) throws Exception
    {
        // The agent lines in order, those of the kind osp taken out for Simulation's parser
        List<String> parsed = new ArrayList<>();
        List<FactoryDescr> osp = new ArrayList<>();
        List<Boolean> isOsp = new ArrayList<>();
        boolean agents = false;
        for (String line : lines)
        {
            if (line.startsWith("~"))
                agents = true;
            if (agents && line.startsWith("Agent="))
            {
                String[] parts = line.substring("Agent=".length()).split(",");
                boolean ours = parts.length > 2 && parts[2].trim().equals("osp");
                isOsp.add(ours);
                if (ours)
                {
                    if (parts.length > 3)
                        throw new IllegalArgumentException("The osp robots take no type flags: " + line);
                    osp.add(new FactoryDescr(new OSPRobotFactory(), parts[1].trim(), Integer.parseInt(parts[0].trim())));
                    continue;
                }
            }
            parsed.add(line);
        }

        Simulation simulation = new Simulation();
        // Simulation.parseConfig, with the whole config kept for its results
        set(simulation, "fullConfig", lines);
        Iterator<String> config = parsed.iterator();
        call(simulation, "parseSuiteConfig", config);
        call(simulation, "readAgentConfiguration", config);
        if ((Boolean) get(simulation, "debug"))
            D.ebug_enable();
        else
            D.ebug_disable();
        // The robots in the config's order
        List<FactoryDescr> theirs = (List<FactoryDescr>) get(simulation, "factories");
        List<FactoryDescr> factories = new ArrayList<>();
        Iterator<FactoryDescr> next = theirs.iterator(), nextOsp = osp.iterator();
        for (boolean ours : isOsp)
            factories.add(ours ? nextOsp.next() : next.next());
        theirs.clear();
        theirs.addAll(factories);
        // The server Simulation would start, as OSPServer
        set(simulation, "practiceServer", new OSPServer(SOCServer.PRACTICE_STRINGPORT, 30, (soc.server.database.DBHelper) get(simulation, "resultsLogger"),
            null, null, (SOCLogger) get(simulation, "gameLogger"), (Boolean) get(simulation, "useParser")));
        return simulation;
    }

    private static Field field(String name) throws NoSuchFieldException
    {
        Field field = Simulation.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }

    private static Object get(Simulation simulation, String name) throws ReflectiveOperationException
    {
        return field(name).get(simulation);
    }

    private static void set(Simulation simulation, String name, Object value) throws ReflectiveOperationException
    {
        field(name).set(simulation, value);
    }

    private static void call(Simulation simulation, String name, Iterator<String> config) throws ReflectiveOperationException, IOException
    {
        Method method = Simulation.class.getDeclaredMethod(name, Iterator.class);
        method.setAccessible(true);
        method.invoke(simulation, config);
    }
}

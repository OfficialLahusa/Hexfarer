package osp.stacsettlers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import soc.game.SOCBoard;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCDevCardSet;
import soc.game.SOCGame;
import soc.game.SOCMoveRobberResult;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.util.IntPair;

/**
 * A StacSettlers game (its SOCGame, JSettlers 1.1.16's, which holds StacSettlers' rules) in the Open Seat Protocol (see
 * ../../PROTOCOL.md): its state as the referee or a seat sees it, the legal moves of the seat that acts, and playing a
 * move with its result. The legal moves are the protocol's moves StacSettlers' own checks allow (SOCGame's and
 * SOCPlayer's can..., could... and isPotential... methods, the ones its server asks before it takes a request), one
 * protocol move where StacSettlers takes two steps: a robber move with its victim, a Year of Plenty or Monopoly card with
 * its resources, a bought road, settlement or city with its place. Its players discard at the same time after a 7; the
 * protocol asks them one after another, from the lowest seat. It keeps no resource bank: the bank is what the hands leave
 * of 19 cards a resource. It draws who starts; the protocol's seats are the turn order from the first player (seat 0
 * starts), so a StacSettlers player number becomes seatOf it.
 */
public final class OSPGame
{
    /**
     * The rules StacSettlers plays, as the protocol names them (see ../../RULESET.md): the rulebook without trades for
     * the seat (Hexfarer's robots trade with the bank and the harbors only; StacSettlers' own agents still trade among
     * themselves in chat), narrower where StacSettlers' engine is: no resource bank (every production is paid), only
     * players with cards robbed, a Road Building card's roads placed at once (the card only with a road left and a place
     * for it). Its bank trades take every rate the player has, and a player wins on their own turn, as in the rulebook.
     */
    public static final Map<String, Object> RULESET = java.util.Collections.unmodifiableMap(Json.object("preset", "standard",
        "player_trading", "none", "bank_shortage", "unlimited", "robber_victims", "with_cards", "road_building", "forced"));

    // The protocol's resource names by StacSettlers' resource type (the same numbers as JSettlers 2), and back
    private static final String[] RESOURCE_NAMES = new String[SOCResourceConstants.WOOD + 1];
    static
    {
        RESOURCE_NAMES[SOCResourceConstants.CLAY] = "brick";
        RESOURCE_NAMES[SOCResourceConstants.ORE] = "ore";
        RESOURCE_NAMES[SOCResourceConstants.SHEEP] = "wool";
        RESOURCE_NAMES[SOCResourceConstants.WHEAT] = "grain";
        RESOURCE_NAMES[SOCResourceConstants.WOOD] = "lumber";
    }
    private static final int[] RESOURCES = {SOCResourceConstants.CLAY, SOCResourceConstants.ORE, SOCResourceConstants.SHEEP,
        SOCResourceConstants.WHEAT, SOCResourceConstants.WOOD};
    private static final String[] CARD_NAMES = {"knight", "road_building", "year_of_plenty", "monopoly", "victory_point"};

    public final SOCGame game;
    public final OSPBoard board;

    // StacSettlers' player number of seat 0
    public final int first;

    public OSPGame(SOCGame game)
    {
        this.game = game;
        this.board = new OSPBoard(game.getBoard());
        this.first = game.getFirstPlayer();
    }

    /** The protocol's seat of a StacSettlers player number: its place in the turn order from the first player */
    public int seatOf(int player)
    {
        return (player - first + game.maxPlayers) % game.maxPlayers;
    }

    /** StacSettlers' player number of a protocol seat */
    public int playerOf(int seat)
    {
        return (seat + first) % game.maxPlayers;
    }

    public static String resourceName(int type)
    {
        return RESOURCE_NAMES[type];
    }

    public static int resourceType(String name)
    {
        for (int type : RESOURCES)
            if (RESOURCE_NAMES[type].equals(name))
                return type;
        throw new IllegalArgumentException("No resource " + name);
    }

    /**
     * The protocol's name of a development card type, by StacSettlers' numbers (JSettlers 1.1.16's: the knight is 0,
     * where JSettlers 2 has 9); every victory point card is one
     */
    public static String cardName(int type)
    {
        switch (type)
        {
        case SOCDevCardConstants.KNIGHT: return "knight";
        case SOCDevCardConstants.ROADS: return "road_building";
        case SOCDevCardConstants.DISC: return "year_of_plenty";
        case SOCDevCardConstants.MONO: return "monopoly";
        case SOCDevCardConstants.CAP: case SOCDevCardConstants.LIB: case SOCDevCardConstants.UNIV: case SOCDevCardConstants.TEMP:
        case SOCDevCardConstants.TOW: return "victory_point";
        default: throw new IllegalArgumentException("No known development card " + type);
        }
    }

    public boolean isOver()
    {
        return game.getGameState() >= SOCGame.OVER;
    }

    /** The seat that acts now: the lowest one that has to discard, else the current player; -1 once the game is over */
    public int acting()
    {
        if (isOver())
            return -1;
        if (game.getGameState() == SOCGame.WAITING_FOR_DISCARDS)
            for (int seat = 0; seat < game.maxPlayers; seat++)
                if (game.getPlayer(playerOf(seat)).getNeedToDiscard())
                    return seat;
        return seatOf(game.getCurrentPlayerNumber());
    }

    /**
     * Whether the game waits for a protocol decision: not in the states between the two steps of one protocol move
     * (a bought piece waiting for its place, a robber waiting for its victim, a card waiting for its resources)
     */
    public boolean deciding()
    {
        switch (game.getGameState())
        {
        case SOCGame.START1A: case SOCGame.START1B: case SOCGame.START2A: case SOCGame.START2B:
        case SOCGame.PLAY: case SOCGame.PLAY1: case SOCGame.WAITING_FOR_DISCARDS: case SOCGame.PLACING_ROBBER:
        case SOCGame.PLACING_FREE_ROAD1: case SOCGame.PLACING_FREE_ROAD2:
            return true;
        default:
            return false;
        }
    }

    /** The protocol's prompt: what the seat that acts has to do */
    public String prompt()
    {
        switch (game.getGameState())
        {
        case SOCGame.START1A: case SOCGame.START2A: return "setup_settlement";
        case SOCGame.START1B: case SOCGame.START2B: return "setup_road";
        case SOCGame.PLAY: return "roll";
        case SOCGame.WAITING_FOR_DISCARDS: return "discard";
        case SOCGame.PLACING_ROBBER: return "move_robber";
        case SOCGame.PLACING_FREE_ROAD1: case SOCGame.PLACING_FREE_ROAD2: return "road_building";
        default: return isOver() ? "over" : "main";
        }
    }

    // The free roads of a Road Building card still to place: StacSettlers gives two (state PLACING_FREE_ROAD1) only with
    // two roads left and two places for them, else one
    private int freeRoads()
    {
        int state = game.getGameState();
        if (state != SOCGame.PLACING_FREE_ROAD1 && state != SOCGame.PLACING_FREE_ROAD2)
            return 0;
        SOCPlayer player = game.getPlayer(game.getCurrentPlayerNumber());
        int free = Math.min(state == SOCGame.PLACING_FREE_ROAD1 ? 2 : 1, player.getNumPieces(SOCPlayingPiece.ROAD));
        return roadSites(player).isEmpty() ? 0 : free;
    }

    // A hand's cards by the protocol's names (every resource, 0 included)
    private static Map<String, Object> cards(SOCResourceSet set)
    {
        Map<String, Object> cards = new LinkedHashMap<>();
        for (int type : RESOURCES)
            cards.put(RESOURCE_NAMES[type], (long) set.getAmount(type));
        return cards;
    }

    // The resource cards of a hand (StacSettlers' sets also count unknown cards, which the true game has none of)
    private static int total(SOCResourceSet set)
    {
        int total = 0;
        for (int type : RESOURCES)
            total += set.getAmount(type);
        return total;
    }

    /**
     * A state's hash as the protocol's decisions carry it: the first 16 hex digits of the SHA-256 of its canonical JSON
     * (keys sorted, no whitespace, UTF-8)
     */
    public static String hash(Object state)
    {
        try
        {
            byte[] digest = java.security.MessageDigest.getInstance("SHA-256").digest(Json.write(state, true).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < 8; i++)
                hex.append(String.format("%02x", digest[i]));
            return hex.toString();
        }
        catch (java.security.NoSuchAlgorithmException e)
        {
            throw new IllegalStateException(e);
        }
    }

    /** A card set of only the cards there, for moves */
    public static Map<String, Object> cardSet(int[] counts)
    {
        Map<String, Object> cards = new LinkedHashMap<>();
        for (int type : RESOURCES)
            if (counts[type] > 0)
                cards.put(RESOURCE_NAMES[type], (long) counts[type]);
        return cards;
    }

    /** A card set of only the cards there, from StacSettlers' resources */
    public static Map<String, Object> cardSet(SOCResourceSet set)
    {
        int[] counts = new int[SOCResourceConstants.WOOD + 1];
        for (int type : RESOURCES)
            counts[type] = set.getAmount(type);
        return cardSet(counts);
    }

    @SuppressWarnings("unchecked")
    public static SOCResourceSet resourceSet(Object set)
    {
        int[] counts = new int[SOCResourceConstants.WOOD + 1];
        for (Map.Entry<String, Object> card : ((Map<String, Object>) set).entrySet())
            counts[resourceType(card.getKey())] += Json.integer(card.getValue());
        return new SOCResourceSet(counts[SOCResourceConstants.CLAY], counts[SOCResourceConstants.ORE], counts[SOCResourceConstants.SHEEP],
            counts[SOCResourceConstants.WHEAT], counts[SOCResourceConstants.WOOD], 0);
    }

    /**
     * The state in the protocol's shape: as the referee sees it (viewer below 0), or as a seat does (other seats'
     * hands, development cards and points with victory point cards null)
     */
    public Map<String, Object> state(int viewer)
    {
        List<Object> settlements = new ArrayList<>(), cities = new ArrayList<>(), roads = new ArrayList<>(), players = new ArrayList<>();
        int[] held = new int[SOCResourceConstants.WOOD + 1];
        SOCPlayer longest = game.getPlayerWithLongestRoad(), largest = game.getPlayerWithLargestArmy();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            SOCPlayer player = game.getPlayer(playerOf(seat));
            for (Object piece : player.getSettlements())
                settlements.add(Json.array((long) seat, board.intersection(((SOCPlayingPiece) piece).getCoordinates())));
            for (Object piece : player.getCities())
                cities.add(Json.array((long) seat, board.intersection(((SOCPlayingPiece) piece).getCoordinates())));
            for (Object piece : player.getRoads())
                roads.add(Json.array((long) seat, board.path(((SOCPlayingPiece) piece).getCoordinates())));
            SOCResourceSet hand = player.getResources();
            for (int type : RESOURCES)
                held[type] += hand.getAmount(type);
            // Development cards, bought this turn or before, by StacSettlers' card numbers
            Map<String, Object> development = new LinkedHashMap<>();
            for (String name : CARD_NAMES)
                development.put(name, 0L);
            SOCDevCardSet cards = player.getDevCards();
            int developmentSize = 0;
            for (int type = SOCDevCardConstants.MIN; type <= SOCDevCardConstants.MAX_KNOWN; type++)
            {
                int count = cards.getAmount(SOCDevCardSet.OLD, type) + cards.getAmount(SOCDevCardSet.NEW, type);
                if (count == 0)
                    continue;
                developmentSize += count;
                development.put(cardName(type), (Long) development.get(cardName(type)) + count);
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("seat", (long) seat);
            entry.put("hand_size", (long) total(hand));
            entry.put("development_size", (long) developmentSize);
            entry.put("public_vp", (long) player.getPublicVP());
            entry.put("vp", (long) player.getTotalVP());
            entry.put("hand", cards(hand));
            entry.put("development", development);
            entry.put("played", Json.object("knight", (long) player.getNumKnights()));
            entry.put("longest_road", (long) player.getLongestRoadLength());
            entry.put("has_longest_road", longest == player);
            entry.put("has_largest_army", largest == player);
            entry.put("pieces_left", Json.object("roads", (long) player.getNumPieces(SOCPlayingPiece.ROAD),
                "settlements", (long) player.getNumPieces(SOCPlayingPiece.SETTLEMENT), "cities", (long) player.getNumPieces(SOCPlayingPiece.CITY)));
            players.add(entry);
        }
        // StacSettlers has no resource bank (its ruleset says unlimited): the counts stop at 0, though it still pays
        Map<String, Object> bank = new LinkedHashMap<>();
        for (int type : RESOURCES)
            bank.put(RESOURCE_NAMES[type], (long) Math.max(0, 19 - held[type]));
        bank.put("development", (long) game.getNumDevCards());

        Map<String, Object> state = new LinkedHashMap<>();
        state.put("prompt", prompt());
        // Who acts: a player discarding after a 7 may be another than the one on turn
        int acting = acting();
        state.put("current", (long) (acting >= 0 ? acting : seatOf(game.getCurrentPlayerNumber())));
        state.put("on_turn", (long) seatOf(game.getCurrentPlayerNumber()));
        state.put("free_roads", (long) freeRoads());
        state.put("robber", board.hex(game.getBoard().getRobberHex()));
        state.put("buildings", Json.object("settlements", settlements, "cities", cities, "roads", roads));
        state.put("bank", bank);
        state.put("players", players);
        state.put("offer", null);
        return viewer < 0 ? state : hide(state, viewer);
    }

    // Where the player may build a road now: every land edge StacSettlers calls a potential road
    private List<Integer> roadSites(SOCPlayer player)
    {
        List<Integer> sites = new ArrayList<>();
        for (int edge : board.edges())
            if (player.isPotentialRoad(edge))
                sites.add(edge);
        return sites;
    }

    /**
     * The seats a robber at the hex could steal from, as StacSettlers' game finds them (getPossibleVictims, for the hex
     * the robber is on): the players with a building there (getPlayersOnHex), not the robber, holding a resource card
     */
    List<Integer> victims(int hex, int robberSeat)
    {
        int robber = playerOf(robberSeat);
        List<Integer> victims = new ArrayList<>();
        for (Object on : game.getPlayersOnHex(hex))
        {
            SOCPlayer player = (SOCPlayer) on;
            int pn = player.getPlayerNumber();
            if (pn != robber && ! game.isSeatVacant(pn) && total(player.getResources()) > 0)
                victims.add(seatOf(pn));
        }
        victims.sort(null);
        return victims;
    }

    private static List<Object> move(String type, Object value)
    {
        return value == null ? Json.array(type) : Json.array(type, value);
    }

    /** The legal moves of the seat (which has to be the one that acts), as StacSettlers' rules allow them */
    public List<Object> legal(int seat)
    {
        List<Object> legal = new ArrayList<>();
        if (seat < 0 || seat != acting() || ! deciding())
            return legal;
        int pn = playerOf(seat);
        SOCPlayer player = game.getPlayer(pn);
        switch (prompt())
        {
        case "setup_settlement":
            for (int node : board.nodes())
                if (player.isPotentialSettlement(node))
                    legal.add(move("build_settlement", board.intersection(node)));
            break;
        case "setup_road":
        {
            // The road goes next to the settlement just placed (StacSettlers' server takes any road its player could build,
            // the rulebook only this one)
            List<?> next = game.getBoard().getAdjacentEdgesToNode(player.getLastSettlementCoord());
            for (int edge : roadSites(player))
                if (next.contains(edge))
                    legal.add(move("build_road", board.path(edge)));
            break;
        }
        case "road_building":
            for (int edge : roadSites(player))
                legal.add(move("build_road", board.path(edge)));
            break;
        case "discard":
            legal.addAll(discardMoves(seat));
            break;
        case "move_robber":
        {
            int robber = game.getBoard().getRobberHex();
            for (int hex : board.landHexes())
            {
                if (hex == robber || ! game.canMoveRobber(pn, hex))
                    continue;
                List<Integer> victims = victims(hex, seat);
                if (victims.isEmpty())
                    legal.add(move("move_robber", Json.object("hex", board.hex(hex), "victim", null)));
                for (int victim : victims)
                    legal.add(move("move_robber", Json.object("hex", board.hex(hex), "victim", (long) victim)));
            }
            break;
        }
        case "roll":
            if (game.canRollDice(pn))
                legal.add(move("roll", null));
            cardMoves(pn, legal);
            break;
        case "main":
            if (game.couldBuildRoad(pn))
                for (int edge : roadSites(player))
                    legal.add(move("build_road", board.path(edge)));
            if (game.couldBuildSettlement(pn))
                for (int node : board.nodes())
                    if (player.isPotentialSettlement(node))
                        legal.add(move("build_settlement", board.intersection(node)));
            if (game.couldBuildCity(pn))
                for (int node : board.nodes())
                    if (player.isPotentialCity(node))
                        legal.add(move("build_city", board.intersection(node)));
            if (game.couldBuyDevCard(pn))
                legal.add(move("buy_development", null));
            cardMoves(pn, legal);
            // Bank trades of one card at every rate the player has: 4:1 always, 3:1 at a generic harbor, 2:1 at the
            // resource's (StacSettlers' canMakeBankTrade takes each)
            for (int give : RESOURCES)
                for (int rate : new int[] {4, 3, 2})
                {
                    if (rate == 3 && ! player.getPortFlag(SOCBoard.MISC_PORT) || rate == 2 && ! player.getPortFlag(give))
                        continue;
                    for (int get : RESOURCES)
                    {
                        if (get == give)
                            continue;
                        int[] gives = new int[SOCResourceConstants.WOOD + 1], gets = new int[SOCResourceConstants.WOOD + 1];
                        gives[give] = rate;
                        gets[get] = 1;
                        SOCResourceSet giving = resourceSet(cardSet(gives));
                        if (player.getResources().contains(giving) && game.canMakeBankTrade(giving, resourceSet(cardSet(gets))))
                            legal.add(move("bank_trade", Json.object("give", cardSet(gives), "get", cardSet(gets))));
                    }
                }
            if (game.canEndTurn(pn))
                legal.add(move("end_turn", null));
            break;
        default:
            break;
        }
        return legal;
    }

    // The development cards the player may play now, Year of Plenty and Monopoly with every choice of resources
    private void cardMoves(int pn, List<Object> legal)
    {
        if (game.canPlayKnight(pn))
            legal.add(move("play_knight", null));
        // As road_building forced has it: only with a road left (StacSettlers checks that) and a place for it (it doesn't,
        // and a card played without one would leave the game waiting for a road nobody can place)
        if (game.canPlayRoadBuilding(pn) && ! roadSites(game.getPlayer(pn)).isEmpty())
            legal.add(move("play_road_building", null));
        if (game.canPlayDiscovery(pn))
            for (int i = 0; i < RESOURCES.length; i++)
                for (int j = i; j < RESOURCES.length; j++)
                {
                    List<Object> pick = Json.array(RESOURCE_NAMES[RESOURCES[i]], RESOURCE_NAMES[RESOURCES[j]]);
                    pick.sort(null);
                    legal.add(move("play_year_of_plenty", pick));
                }
        if (game.canPlayMonopoly(pn))
            for (int type : RESOURCES)
                legal.add(move("play_monopoly", RESOURCE_NAMES[type]));
    }

    /** The discards of a seat that has to discard: every set of half its cards (rounded down) */
    public List<Object> discardMoves(int seat)
    {
        SOCPlayer player = game.getPlayer(playerOf(seat));
        List<Object> legal = new ArrayList<>();
        if (player.getNeedToDiscard())
            discards(player.getResources(), total(player.getResources()) / 2, legal);
        return legal;
    }

    /** A state as a seat sees it: the other seats' hands, development cards and points with them hidden (null) */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> hide(Map<String, Object> state, int viewer)
    {
        Map<String, Object> view = copy(state);
        for (Object entry : (List<Object>) view.get("players"))
        {
            Map<String, Object> player = (Map<String, Object>) entry;
            if (Json.integer(player.get("seat")) != viewer)
                for (String field : new String[] {"vp", "hand", "development"})
                    player.put(field, null);
        }
        return view;
    }

    /** A deep copy of a protocol value */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> copy(Map<String, Object> value)
    {
        return (Map<String, Object>) Json.read(Json.write(value));
    }

    // Every card set of the count the hand can give up
    private static void discards(SOCResourceSet hand, int count, List<Object> legal)
    {
        int[] amounts = new int[SOCResourceConstants.WOOD + 1];
        discards(hand, count, 0, amounts, legal);
    }

    private static void discards(SOCResourceSet hand, int left, int index, int[] amounts, List<Object> legal)
    {
        if (index == RESOURCES.length)
        {
            if (left == 0)
                legal.add(move("discard", cardSet(amounts)));
            return;
        }
        int type = RESOURCES[index];
        for (int n = 0; n <= Math.min(left, hand.getAmount(type)); n++)
        {
            amounts[type] = n;
            discards(hand, left - n, index + 1, amounts, legal);
        }
        amounts[type] = 0;
    }

    // A move StacSettlers' server wouldn't take
    private static void require(boolean check, String what)
    {
        if (! check)
            throw new IllegalStateException("StacSettlers' server wouldn't take " + what);
    }

    /**
     * Plays a legal move of the seat by StacSettlers' rules, each step after the check its server makes before it takes
     * the request (else IllegalStateException), and returns its result as the referee knows it: the dice, the card bought,
     * the card stolen (null if none), the cards a monopoly took from each other seat; null for other moves. A robber move
     * also checks that the victims the move's hex offered are the ones StacSettlers' game finds there.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(List<Object> move, int seat)
    {
        String type = (String) move.get(0);
        Object value = move.size() > 1 ? move.get(1) : null;
        int pn = playerOf(seat);
        SOCPlayer player = game.getPlayer(pn);
        SOCBoard b = game.getBoard();
        int state = game.getGameState();
        boolean initial = state < SOCGame.PLAY;
        switch (type)
        {
        case "roll":
        {
            require(game.canRollDice(pn), "the roll");
            IntPair roll = game.rollDice();
            return Json.object("dice", Json.array((long) roll.getA(), (long) roll.getB()));
        }
        case "build_settlement":
        {
            int node = board.node(value);
            if (! initial)
            {
                require(state == SOCGame.PLAY1 && game.couldBuildSettlement(pn), "buying a settlement");
                game.buySettlement(pn);
            }
            require(player.isPotentialSettlement(node), "a settlement at " + Json.write(value));
            game.putPiece(new SOCSettlement(player, node, b));
            return null;
        }
        case "build_road":
        {
            int edge = board.edge(value);
            if (! initial && state != SOCGame.PLACING_FREE_ROAD1 && state != SOCGame.PLACING_FREE_ROAD2)
            {
                require(state == SOCGame.PLAY1 && game.couldBuildRoad(pn), "buying a road");
                game.buyRoad(pn);
            }
            require(player.isPotentialRoad(edge), "a road at " + Json.write(value));
            game.putPiece(new SOCRoad(player, edge, b));
            return null;
        }
        case "build_city":
        {
            int node = board.node(value);
            require(state == SOCGame.PLAY1 && game.couldBuildCity(pn), "buying a city");
            game.buyCity(pn);
            require(player.isPotentialCity(node), "a city at " + Json.write(value));
            game.putPiece(new SOCCity(player, node, b));
            return null;
        }
        case "buy_development":
            require((state == SOCGame.PLAY1) && game.couldBuyDevCard(pn), "buying a development card");
            return Json.object("card", cardName(game.buyDevCard()));
        case "play_knight":
            require(game.canPlayKnight(pn), "a knight");
            game.playKnight();
            return null;
        case "play_road_building":
            require(game.canPlayRoadBuilding(pn), "a Road Building card");
            game.playRoadBuilding();
            return null;
        case "play_year_of_plenty":
        {
            require(game.canPlayDiscovery(pn), "a Year of Plenty card");
            game.playDiscovery();
            int[] counts = new int[SOCResourceConstants.WOOD + 1];
            for (Object name : (List<Object>) value)
                counts[resourceType((String) name)]++;
            SOCResourceSet pick = resourceSet(cardSet(counts));
            require(game.canDoDiscoveryAction(pick), "the pick " + Json.write(value));
            game.doDiscoveryAction(pick);
            return null;
        }
        case "play_monopoly":
        {
            require(game.canPlayMonopoly(pn), "a Monopoly card");
            game.playMonopoly();
            require(game.canDoMonopolyAction(), "the monopoly of " + value);
            int[] taken = game.doMonopolyAction(resourceType((String) value));
            Map<String, Object> result = new LinkedHashMap<>();
            for (int other = 0; other < game.maxPlayers; other++)
                if (other != seat)
                    result.put(Integer.toString(other), (long) taken[playerOf(other)]);
            return Json.object("taken", result);
        }
        case "move_robber":
        {
            Map<String, Object> robbery = (Map<String, Object>) value;
            int hex = board.hexCoord((List<?>) robbery.get("hex"));
            require(game.canMoveRobber(pn, hex), "the robber at " + Json.write(robbery.get("hex")));
            List<Integer> offered = victims(hex, seat);
            SOCMoveRobberResult result = game.moveRobber(pn, hex);
            List<Integer> found = new ArrayList<>();
            for (Object victim : result.getVictims())
                found.add(seatOf(((SOCPlayer) victim).getPlayerNumber()));
            found.sort(null);
            require(found.equals(offered), "the victims " + offered + " at " + Json.write(robbery.get("hex")) + ": its game finds " + found);
            int loot = result.getLoot();
            if (game.getGameState() == SOCGame.WAITING_FOR_CHOICE)
            {
                int victim = playerOf(Json.integer(robbery.get("victim")));
                require(game.canChoosePlayer(victim), "the victim " + robbery.get("victim"));
                loot = game.stealFromPlayer(victim);
            }
            return Json.object("stolen", loot > 0 ? resourceName(loot) : null);
        }
        case "discard":
        {
            SOCResourceSet set = resourceSet(value);
            require(game.canDiscard(pn, set), "the discard " + Json.write(value));
            game.discard(pn, set);
            return null;
        }
        case "bank_trade":
        {
            Map<String, Object> trade = (Map<String, Object>) value;
            SOCResourceSet give = resourceSet(trade.get("give")), get = resourceSet(trade.get("get"));
            require(game.canMakeBankTrade(give, get), "the bank trade " + Json.write(value));
            game.makeBankTrade(give, get);
            return null;
        }
        case "end_turn":
            require(game.canEndTurn(pn), "the end of the turn");
            game.endTurn();
            return null;
        default:
            throw new IllegalArgumentException("No StacSettlers move for " + type);
        }
    }
}
